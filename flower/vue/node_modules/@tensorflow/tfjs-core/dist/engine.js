/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { KernelBackend } from './backends/backend';
import { Environment, setEnvironmentGlobal } from './environment';
import { getGlobalNamespace } from './global_util';
import { Add, Cast, Identity } from './kernel_names';
import { getGradient, getKernel, getKernelsForBackend } from './kernel_registry';
import * as log from './log';
import { Profiler } from './profiler';
import { backpropagateGradients, getFilteredNodesXToY } from './tape';
import { setTensorTracker, Tensor, Variable } from './tensor';
import { getTensorsInContainer } from './tensor_util';
import * as util from './util';
import { bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape } from './util';
function isRegisteredKernelInvocation(kernelInvocation) {
    return kernelInvocation.kernelName != null;
}
class EngineState {
    constructor() {
        // Public since optimizers will use it.
        this.registeredVariables = {};
        this.nextTapeNodeId = 0;
        this.numBytes = 0;
        this.numTensors = 0;
        this.numStringTensors = 0;
        this.numDataBuffers = 0;
        // Number of nested tf.grad() statements when computing higher-order
        // gradients. E.g. `1` for first-order gradients and `2` for second-order
        // gradients. Used to track if the tape should be removed after a backprop.
        this.gradientDepth = 0;
        // Number of nested kernel calls. When kernel depth is greater than 1, we turn
        // off the tape.
        this.kernelDepth = 0;
        this.scopeStack = [];
        /**
         * Keeps track of the number of data moves during a kernel execution. We
         * maintain a stack since kernels can call other kernels, recursively.
         */
        this.numDataMovesStack = [];
        this.nextScopeId = 0;
        this.tensorInfo = new WeakMap();
        this.profiling = false;
        this.activeProfile = {
            newBytes: 0,
            newTensors: 0,
            peakBytes: 0,
            kernels: [],
            result: null,
            get kernelNames() {
                return Array.from(new Set(this.kernels.map(k => k.name)));
            }
        };
    }
    dispose() {
        for (const variableName in this.registeredVariables) {
            this.registeredVariables[variableName].dispose();
        }
    }
}
class Engine {
    constructor(ENV) {
        this.ENV = ENV;
        this.registry = {};
        this.registryFactory = {};
        this.pendingBackendInitId = 0;
        this.state = new EngineState();
    }
    async ready() {
        if (this.pendingBackendInit != null) {
            return this.pendingBackendInit.then(() => { });
        }
        if (this.backendInstance != null) {
            return;
        }
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const success = await this.initializeBackend(backendName).success;
            if (success) {
                await this.setBackend(backendName);
                return;
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    get backend() {
        if (this.pendingBackendInit != null) {
            throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make ` +
                `sure to await tf.ready() or await tf.setBackend() before calling ` +
                `other methods`);
        }
        if (this.backendInstance == null) {
            const { name, asyncInit } = this.initializeBackendsAndReturnBest();
            if (asyncInit) {
                throw new Error(`The highest priority backend '${name}' has not yet been ` +
                    `initialized. Make sure to await tf.ready() or ` +
                    `await tf.setBackend() before calling other methods`);
            }
            this.setBackend(name);
        }
        return this.backendInstance;
    }
    backendNames() {
        return Object.keys(this.registryFactory);
    }
    findBackend(backendName) {
        if (!(backendName in this.registry)) {
            // If the backend hasn't been initialized but we have a registry entry for
            // it, initialize it and return it.
            if (backendName in this.registryFactory) {
                const { asyncInit } = this.initializeBackend(backendName);
                if (asyncInit) {
                    // Backend is not ready yet.
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return this.registry[backendName];
    }
    findBackendFactory(backendName) {
        if (!(backendName in this.registryFactory)) {
            return null;
        }
        return this.registryFactory[backendName].factory;
    }
    registerBackend(backendName, factory, priority = 1) {
        if (backendName in this.registryFactory) {
            log.warn(`${backendName} backend was already registered. ` +
                `Reusing existing backend factory.`);
            return false;
        }
        this.registryFactory[backendName] = { factory, priority };
        return true;
    }
    async setBackend(backendName) {
        if (this.registryFactory[backendName] == null) {
            throw new Error(`Backend name '${backendName}' not found in registry`);
        }
        this.backendName = backendName;
        if (this.registry[backendName] == null) {
            this.backendInstance = null;
            const { success, asyncInit } = this.initializeBackend(backendName);
            const result = asyncInit ? await success : success;
            if (!result) {
                return false;
            }
        }
        this.backendInstance = this.registry[backendName];
        this.setupRegisteredKernels();
        // Reset the profiler.
        this.profiler = new Profiler(this.backendInstance);
        return true;
    }
    setupRegisteredKernels() {
        const kernels = getKernelsForBackend(this.backendName);
        kernels.forEach(kernel => {
            if (kernel.setupFunc != null) {
                kernel.setupFunc(this.backendInstance);
            }
        });
    }
    disposeRegisteredKernels(backendName) {
        const kernels = getKernelsForBackend(backendName);
        kernels.forEach(kernel => {
            if (kernel.disposeFunc != null) {
                kernel.disposeFunc(this.registry[backendName]);
            }
        });
    }
    /**
     * Initializes a backend by looking up the backend name in the factory
     * registry and calling the factory method. Returns a boolean representing
     * whether the initialization of the backend suceeded. Throws an error if
     * there is no backend in the factory registry.
     */
    initializeBackend(backendName) {
        const registryFactoryEntry = this.registryFactory[backendName];
        if (registryFactoryEntry == null) {
            throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
        }
        try {
            const backend = registryFactoryEntry.factory();
            /* Test if the factory returns a promise.
            Done in a more liberal way than
            previous 'Promise.resolve(backend)===backend'
            as we needed to account for custom Promise
            implementations (e.g. Angular) */
            if (backend && !(backend instanceof KernelBackend) &&
                typeof backend.then === 'function') {
                const promiseId = ++this.pendingBackendInitId;
                const success = backend
                    .then(backendInstance => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.registry[backendName] = backendInstance;
                    this.pendingBackendInit = null;
                    return true;
                })
                    .catch(err => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.pendingBackendInit = null;
                    log.warn(`Initialization of backend ${backendName} failed`);
                    log.warn(err.stack || err.message);
                    return false;
                });
                this.pendingBackendInit = success;
                return { success, asyncInit: true };
            }
            else {
                this.registry[backendName] = backend;
                return { success: true, asyncInit: false };
            }
        }
        catch (err) {
            log.warn(`Initialization of backend ${backendName} failed`);
            log.warn(err.stack || err.message);
            return { success: false, asyncInit: false };
        }
    }
    removeBackend(backendName) {
        if (!(backendName in this.registryFactory)) {
            throw new Error(`${backendName} backend not found in registry`);
        }
        if (this.backendName === backendName && this.pendingBackendInit != null) {
            // There is a pending promise of the backend we want to remove. Make it
            // obsolete.
            this.pendingBackendInitId++;
        }
        if (backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        delete this.registryFactory[backendName];
        // Unset the backend if it is active.
        if (this.backendName === backendName) {
            this.pendingBackendInit = null;
            this.backendName = null;
            this.backendInstance = null;
        }
    }
    getSortedBackends() {
        if (Object.keys(this.registryFactory).length === 0) {
            throw new Error('No backend found in registry.');
        }
        return Object.keys(this.registryFactory).sort((a, b) => {
            // Highest priority comes first.
            return this.registryFactory[b].priority -
                this.registryFactory[a].priority;
        });
    }
    initializeBackendsAndReturnBest() {
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const { success, asyncInit } = this.initializeBackend(backendName);
            if (asyncInit || success) {
                return { name: backendName, asyncInit };
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    moveData(backend, dataId) {
        const info = this.state.tensorInfo.get(dataId);
        const srcBackend = info.backend;
        const values = this.readSync(dataId);
        const refCount = srcBackend.refCount(dataId);
        // Delete the tensor from the old backend and move it to the new
        // backend.
        srcBackend.disposeData(dataId, true);
        info.backend = backend;
        backend.move(dataId, values, info.shape, info.dtype, refCount);
        if (this.shouldCheckForMemLeaks()) {
            // Track the number of moves during a kernel execution to correctly
            // detect memory leaks.
            this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
        }
    }
    tidy(nameOrFn, fn) {
        let name = null;
        if (fn == null) {
            // Called with only 1 argument.
            if (typeof nameOrFn !== 'function') {
                throw new Error('Please provide a function to tidy()');
            }
            fn = nameOrFn;
        }
        else {
            // Called with 2 arguments.
            if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to tidy() must be a string');
            }
            if (typeof fn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to tidy() must be a function');
            }
            name = nameOrFn;
            // TODO(nsthorat,smilkov): Do operation logging and performance
            // profiling.
        }
        let result;
        return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
            result = fn();
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            return result;
        });
    }
    scopedRun(start, end, f) {
        start();
        try {
            const res = f();
            end();
            return res;
        }
        catch (ex) {
            end();
            throw ex;
        }
    }
    nextTensorId() {
        return Engine.nextTensorId++;
    }
    nextVariableId() {
        return Engine.nextVariableId++;
    }
    /**
     * This method is called instead of the public-facing tensor.clone() when
     * saving a tensor for backwards pass. It makes sure to add the clone
     * operation to the tape regardless of being called inside a kernel
     * execution.
     */
    clone(x) {
        const y = ENGINE.runKernel(Identity, { x });
        const inputs = { x };
        const grad = (dy) => ({
            x: () => {
                const dtype = 'float32';
                const gradInputs = { x: dy };
                const attrs = { dtype };
                return ENGINE.runKernel(Cast, gradInputs, 
                // tslint:disable-next-line: no-unnecessary-type-assertion
                attrs);
            }
        });
        const saved = [];
        this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
        return y;
    }
    /**
     * Execute a kernel with the given name and return the output tensor.
     *
     * @param kernelName The name of the kernel to execute.
     * @param inputs A map of input names to tensors.
     * @param attrs A map of attribute names to their values. An attribute is a
     *     primitive (non-tensor) input to the kernel.
     * @param inputsToSave A list of tensors, inputs to save for the backprop
     *     computation.
     * @param outputsToSave A list of booleans, specifying which output to save
     *     for the backprop computation. These are booleans since the output
     * tensors are not visible to the user.
     */
    runKernel(kernelName, inputs, attrs) {
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        const hasKernel = getKernel(kernelName, this.backendName) != null;
        if (!hasKernel) {
            throw new Error(`Kernel '${kernelName}' not registered for backend '${this.backendName}'`);
        }
        return this.runKernelFunc({ kernelName, inputs, attrs });
    }
    shouldCheckForMemLeaks() {
        return this.ENV.getBool('IS_TEST');
    }
    checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
        const numDataIdsAfter = this.backend.numDataIds();
        // Count the number of data ids associated with the result of the kernel.
        let numOutputDataIds = 0;
        outInfos.forEach(info => {
            // Complex numbers allocate 3 data ids, one for 'real', one for
            // 'imaginary', and one for the container that holds the former two.
            numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
        });
        // Account for the number of moves during kernel execution. A "data move"
        // can happen in the middle of a kernel execution, placing a new (key,value)
        // pair in the data storage. Since data moves have net zero effect (we
        // always remove the data from the old backend), we have to cancel them out
        // when detecting memory leaks.
        const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
        const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
        if (dataIdsLeaked > 0) {
            throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
        }
    }
    /**
     * Internal helper method to execute a kernel Func
     *
     * Use `runKernel` to execute kernels from outside of engine.
     */
    runKernelFunc(kernelParams) {
        let outputs;
        let saved = [];
        const isTapeOn = this.isTapeOn();
        const startingBytecount = this.state.numBytes;
        const startingNumTensors = this.state.numTensors;
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack.push(0);
        }
        let kernelFunc;
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        let out;
        const kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
            kernelParams.kernelName :
            this.state.activeScope != null ? this.state.activeScope.name : '';
        // Create the kernelFunc from either a registered kernel OR passed in
        // forward/backward functions (used by custom grad). In this context a
        // kernelFunc wraps a kernel implementation with some bookkeeping.
        if (isRegisteredKernelInvocation(kernelParams)) {
            const { kernelName, inputs, attrs } = kernelParams;
            if (this.backendName == null) {
                // backend has not been initialized yet (backend initialization is lazy
                // can be deferred until an op/ kernel is run).
                // The below getter has side effects that will try to initialize the
                // backend and set properties like this.backendName
                // tslint:disable-next-line: no-unused-expression
                this.backend;
            }
            const kernel = getKernel(kernelName, this.backendName);
            util.assert(kernel != null, () => `Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'`);
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                const outInfos = Array.isArray(out) ? out : [out];
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                }
                const outTensors = outInfos.map((outInfo) => {
                    // todo (yassogba) remove this option (Tensor) when node backend
                    // methods have been modularized and they all return tensorInfo.
                    // TensorInfos do not have a rank attribute.
                    if (outInfo.rank != null) {
                        return outInfo;
                    }
                    return this.makeTensorFromTensorInfo(outInfo);
                });
                // Save any required inputs and outputs.
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since there would be no backprop for these tensors
                // (which would otherwise dispose them).
                if (isTapeOn) {
                    const tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                    saved = this.saveTensorsForBackwardMode(tensorsToSave);
                }
                return outTensors;
            };
        }
        else {
            const { forwardFunc } = kernelParams;
            // Running a customGrad op.
            const saveFunc = (tensors) => {
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since we would never run backprop, which disposes
                // the kept tensors.
                if (!isTapeOn) {
                    return;
                }
                saved = tensors.map(tensor => this.keep(this.clone(tensor)));
            };
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                const outs = (Array.isArray(out) ? out : [out]);
                if (this.shouldCheckForMemLeaks()) {
                    // Scope name is used to print a more helpful error message if needed.
                    this.checkKernelForMemLeak(kernelOrScopeName, numDataIdsBefore, outs);
                }
                return outs;
            };
        }
        //
        // Run the kernelFunc. Optionally profiling it.
        //
        const { inputs, attrs } = kernelParams;
        const backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
            null :
            kernelParams.backwardsFunc;
        let kernelProfile;
        this.scopedRun(
        // Stop recording to a tape when running a kernel.
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
            if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
                outputs = kernelFunc();
            }
            else {
                kernelProfile = this.profiler.profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
                if (this.ENV.getBool('DEBUG')) {
                    this.profiler.logKernelProfile(kernelProfile);
                }
                outputs = kernelProfile.outputs;
            }
        });
        if (isTapeOn) {
            this.addTapeNode(kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
        }
        if (this.state.profiling) {
            this.state.activeProfile.kernels.push({
                name: kernelOrScopeName,
                bytesAdded: this.state.numBytes - startingBytecount,
                totalBytesSnapshot: this.state.numBytes,
                tensorsAdded: this.state.numTensors - startingNumTensors,
                totalTensorsSnapshot: this.state.numTensors,
                inputShapes: Object.keys(inputs).map(key => inputs[key] != null ? inputs[key].shape : null),
                outputShapes: outputs.map(item => item.shape),
                kernelTimeMs: kernelProfile.timeMs,
                extraInfo: kernelProfile.extraInfo
            });
        }
        return (Array.isArray(out) ? outputs : outputs[0]);
    }
    /**
     * Saves tensors used in forward mode for use in backward mode.
     *
     * @param tensors the list of tensors to save.
     */
    saveTensorsForBackwardMode(tensors) {
        const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
        return saved;
    }
    /**
     * Returns a list of tensors to save for a given gradient calculation.
     *
     * @param kernelName name of kernel to look up gradient for.
     * @param inputs a map of input tensors.
     * @param outputs an array of output tensors from forward mode of kernel.
     */
    getTensorsForGradient(kernelName, inputs, outputs) {
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            const inputsToSave = gradConfig.inputsToSave || [];
            const outputsToSave = gradConfig.outputsToSave || [];
            // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
            // specified in inputsToSave will be saved.
            let inputTensorsToSave;
            if (gradConfig.saveAllInputs) {
                util.assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
            }
            else {
                inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
            }
            const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
            return inputTensorsToSave.concat(outputTensorsToSave);
        }
        // We return an empty list rather than throw an error because the kernel we
        // are looking up may not actually be relevant to backproping through the
        // overall function
        //
        // See 'does not error if irrelevant (pruned) ops are missing grads' test
        // in gradients_test.ts for an example.
        return [];
    }
    /**
     * Internal method used by public APIs for tensor creation. Makes a new
     * tensor with the provided shape, dtype and values. It always
     * creates a new data id and writes the values to the underlying backend.
     */
    makeTensor(values, shape, dtype, backend) {
        if (values == null) {
            throw new Error('Values passed to engine.makeTensor() are null');
        }
        dtype = dtype || 'float32';
        backend = backend || this.backend;
        let backendVals = values;
        if (dtype === 'string' && util.isString(values[0])) {
            backendVals = values.map(d => util.encodeString(d));
        }
        const dataId = backend.write(backendVals, shape, dtype);
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        // Count bytes for string tensors.
        if (dtype === 'string') {
            const info = this.state.tensorInfo.get(dataId);
            const newBytes = bytesFromStringArray(backendVals);
            this.state.numBytes += newBytes - info.bytes;
            info.bytes = newBytes;
        }
        return t;
    }
    /**
     * Internal method used by backends. Makes a new tensor
     * that is a wrapper around an existing data id. It doesn't create
     * a new data id, only increments the ref count used in memory tracking.
     * @deprecated
     */
    makeTensorFromDataId(dataId, shape, dtype, backend) {
        dtype = dtype || 'float32';
        const tensorInfo = { dataId, shape, dtype };
        return this.makeTensorFromTensorInfo(tensorInfo, backend);
    }
    /**
     * Internal method used by backends. Makes a new tensor that is a wrapper
     * around an existing data id in TensorInfo. It doesn't create a new data id,
     * only increments the ref count used in memory tracking.
     */
    makeTensorFromTensorInfo(tensorInfo, backend) {
        const { dataId, shape, dtype } = tensorInfo;
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        return t;
    }
    makeVariable(initialValue, trainable = true, name, dtype) {
        name = name || this.nextVariableId().toString();
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.cast(dtype);
        }
        const v = new Variable(initialValue, trainable, name, this.nextTensorId());
        if (this.state.registeredVariables[v.name] != null) {
            throw new Error(`Variable with name ${v.name} was already registered`);
        }
        this.state.registeredVariables[v.name] = v;
        this.incRef(v, this.backend);
        return v;
    }
    trackTensor(a, backend) {
        this.state.numTensors++;
        if (a.dtype === 'string') {
            this.state.numStringTensors++;
        }
        // Bytes for complex numbers are counted by their components. Bytes for
        // string tensors are counted when writing values.
        let bytes = 0;
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            bytes = a.size * util.bytesPerElement(a.dtype);
        }
        this.state.numBytes += bytes;
        if (!this.state.tensorInfo.has(a.dataId)) {
            this.state.numDataBuffers++;
            this.state.tensorInfo.set(a.dataId, {
                backend: backend || this.backend,
                dtype: a.dtype,
                shape: a.shape,
                bytes
            });
        }
        if (!(a instanceof Variable)) {
            this.track(a);
        }
    }
    // Track the tensor by dataId and increase the refCount for the dataId in the
    // backend.
    // TODO(pyu10055): This is currently used by makeVariable method, to increase
    // refCount on the backend for the dataId. It can potentially be replaced with
    // Identity op indead of calling backend directly.
    incRef(a, backend) {
        this.trackTensor(a, backend);
        this.backend.incRef(a.dataId);
    }
    removeDataId(dataId, backend) {
        if (this.state.tensorInfo.has(dataId) &&
            this.state.tensorInfo.get(dataId).backend === backend) {
            this.state.tensorInfo.delete(dataId);
            this.state.numDataBuffers--;
        }
    }
    disposeTensor(a) {
        if (!this.state.tensorInfo.has(a.dataId)) {
            return;
        }
        const info = this.state.tensorInfo.get(a.dataId);
        this.state.numTensors--;
        if (a.dtype === 'string') {
            this.state.numStringTensors--;
            this.state.numBytes -= info.bytes;
        }
        // Don't count bytes for complex numbers as they are counted by their
        // components.
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            const bytes = a.size * util.bytesPerElement(a.dtype);
            this.state.numBytes -= bytes;
        }
        // Remove the reference to dataId if backend dispose the data successfully
        if (info.backend.disposeData(a.dataId)) {
            this.removeDataId(a.dataId, info.backend);
        }
        // TODO(nsthorat): Construct an error and save the stack trace for
        // debugging when in debug mode. Creating a stack trace is too expensive
        // to do unconditionally.
    }
    disposeVariables() {
        for (const varName in this.state.registeredVariables) {
            const v = this.state.registeredVariables[varName];
            this.disposeVariable(v);
        }
    }
    disposeVariable(v) {
        this.disposeTensor(v);
        if (this.state.registeredVariables[v.name] != null) {
            delete this.state.registeredVariables[v.name];
        }
    }
    memory() {
        const info = this.backend.memory();
        info.numTensors = this.state.numTensors;
        info.numDataBuffers = this.state.numDataBuffers;
        info.numBytes = this.state.numBytes;
        if (this.state.numStringTensors > 0) {
            info.unreliable = true;
            if (info.reasons == null) {
                info.reasons = [];
            }
            info.reasons.push('Memory usage by string tensors is approximate ' +
                '(2 bytes per character)');
        }
        return info;
    }
    async profile(query) {
        this.state.profiling = true;
        const startBytes = this.state.numBytes;
        const startNumTensors = this.state.numTensors;
        this.state.activeProfile.kernels = [];
        this.state.activeProfile.result = await query();
        this.state.profiling = false;
        this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
        this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
        this.state.activeProfile.newTensors =
            this.state.numTensors - startNumTensors;
        for (const kernel of this.state.activeProfile.kernels) {
            kernel.kernelTimeMs = await kernel.kernelTimeMs;
            kernel.extraInfo = await kernel.extraInfo;
        }
        return this.state.activeProfile;
    }
    isTapeOn() {
        return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
    }
    addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
        const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            gradientsFunc = gradConfig.gradFunc;
        }
        if (gradientsFunc != null) {
            tapeNode.gradient = (dys) => {
                // TODO(smilkov): To optimize back-prop, pass dys that are not used in
                // the backprop graph to the user as null instead of zeros
                dys = dys.map((dy, i) => {
                    if (dy == null) {
                        const output = outputs[i];
                        const vals = util.makeZerosTypedArray(output.size, output.dtype);
                        return this.makeTensor(vals, output.shape, output.dtype);
                    }
                    return dy;
                });
                // Grad functions of ops with single outputs expect a dy, while ops
                // with multiple outputs expect dys (array of dy).
                return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
            };
        }
        this.state.activeTape.push(tapeNode);
    }
    keep(result) {
        result.kept = true;
        return result;
    }
    startTape() {
        if (this.state.gradientDepth === 0) {
            this.state.activeTape = [];
        }
        this.state.gradientDepth++;
    }
    endTape() {
        this.state.gradientDepth--;
    }
    /**
     * Start a scope. Use this with endScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    startScope(name) {
        const scopeInfo = {
            track: [],
            name: 'unnamed scope',
            id: this.state.nextScopeId++
        };
        if (name) {
            scopeInfo.name = name;
        }
        this.state.scopeStack.push(scopeInfo);
        this.state.activeScope = scopeInfo;
    }
    /**
     * End a scope. Use this with startScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    endScope(result) {
        const tensorsToTrackInParent = getTensorsInContainer(result);
        const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
        // Dispose the arrays tracked in this scope.
        for (let i = 0; i < this.state.activeScope.track.length; i++) {
            const tensor = this.state.activeScope.track[i];
            if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                tensor.dispose();
            }
        }
        const oldScope = this.state.scopeStack.pop();
        this.state.activeScope = this.state.scopeStack.length === 0 ?
            null :
            this.state.scopeStack[this.state.scopeStack.length - 1];
        // Track the current result in the parent scope.
        tensorsToTrackInParent.forEach(tensor => {
            // Only track the tensor if was allocated in the inner scope and is not
            // globally kept.
            if (!tensor.kept && tensor.scopeId === oldScope.id) {
                this.track(tensor);
            }
        });
    }
    /**
     * Returns gradients of `f` with respect to each of the `xs`. The gradients
     * returned are of the same length as `xs`, but some might be null if `f`
     * was not a function of that `x`. It also takes optional dy to multiply the
     * gradient, which defaults to `1`.
     */
    gradients(f, xs, dy, allowNoGradients = false) {
        util.assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
        if (dy != null && dy.dtype !== 'float32') {
            throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
        }
        const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
        util.assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
        // Filter out the nodes that don't connect x => y.
        const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
        if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
            throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                'that the f you passed encloses all operations that lead from x ' +
                'to y.');
        }
        return this.tidy('backward', () => {
            const accumulatedGradientMap = {};
            accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
            // Backprop gradients through the filtered nodes.
            backpropagateGradients(accumulatedGradientMap, filteredTape, 
            // Pass the tidy function to avoid circular dep with `tape.ts`.
            f => this.tidy(f), 
            // Pass an add function to avoide a circular dep with `tape.ts`.
            add);
            const grads = xs.map(x => accumulatedGradientMap[x.id]);
            if (this.state.gradientDepth === 0) {
                // This means that we are not computing higher-order gradients
                // and can clean up the tape.
                this.state.activeTape.forEach(node => {
                    for (const tensor of node.saved) {
                        tensor.dispose();
                    }
                });
                this.state.activeTape = null;
            }
            return { value: y, grads };
        });
    }
    customGrad(f) {
        util.assert(util.isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
        return (...inputs) => {
            util.assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                'tensors');
            let res;
            const inputMap = {};
            inputs.forEach((input, i) => {
                inputMap[i] = input;
            });
            const forwardFunc = (_, save) => {
                res = f(...[...inputs, save]);
                util.assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.value` is a tensor');
                util.assert(util.isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function.');
                return res.value;
            };
            const backwardsFunc = (dy, saved) => {
                const gradRes = res.gradFunc(dy, saved);
                const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                util.assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
                util.assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'a list of only tensors.');
                const gradMap = {};
                grads.forEach((grad, i) => {
                    gradMap[i] = () => grad;
                });
                return gradMap;
            };
            return this.runKernelFunc({
                forwardFunc,
                backwardsFunc,
                inputs: inputMap,
            });
        };
    }
    readSync(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readSync(dataId);
    }
    read(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.read(dataId);
    }
    readToGPU(dataId, options) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readToGPU(dataId, options);
    }
    async time(query) {
        const start = now();
        const timingInfo = await this.backend.time(query);
        timingInfo.wallMs = now() - start;
        return timingInfo;
    }
    /**
     * Tracks a Tensor in the current scope to be automatically cleaned up
     * when the current scope ends, and returns the value.
     *
     * @param result The Tensor to track in the current scope.
     */
    track(result) {
        if (this.state.activeScope != null) {
            result.scopeId = this.state.activeScope.id;
            this.state.activeScope.track.push(result);
        }
        return result;
    }
    get registeredVariables() {
        return this.state.registeredVariables;
    }
    /**
     * Resets the engine state. Removes all backends but does not remove
     * registered backend factories.
     */
    reset() {
        // Make any pending promise obsolete.
        this.pendingBackendInitId++;
        this.state.dispose();
        this.ENV.reset();
        this.state = new EngineState();
        for (const backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        this.backendName = null;
        this.backendInstance = null;
        this.pendingBackendInit = null;
    }
}
Engine.nextTensorId = 0;
Engine.nextVariableId = 0;
export { Engine };
function ones(shape) {
    const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
    return ENGINE.makeTensor(values, shape, 'float32');
}
export function getOrMakeEngine() {
    const ns = getGlobalNamespace();
    if (ns._tfengine == null) {
        const environment = new Environment(ns);
        ns._tfengine = new Engine(environment);
    }
    setEnvironmentGlobal(ns._tfengine.ENV);
    // Tell the current tensor interface that the global engine is responsible
    // for tracking.
    setTensorTracker(() => ns._tfengine);
    return ns._tfengine;
}
export const ENGINE = getOrMakeEngine();
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
export function add(a, b) {
    // We duplicate Add here to avoid a circular dependency with add.ts.
    const inputs = { a, b };
    return ENGINE.runKernel(Add, inputs);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5naW5lLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9lbmdpbmUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUErQixhQUFhLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUMvRSxPQUFPLEVBQUMsV0FBVyxFQUFFLG9CQUFvQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQ2hFLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUNqRCxPQUFPLEVBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNuRCxPQUFPLEVBQUUsV0FBVyxFQUFFLFNBQVMsRUFBRSxvQkFBb0IsRUFBMEIsTUFBTSxtQkFBbUIsQ0FBQztBQUV6RyxPQUFPLEtBQUssR0FBRyxNQUFNLE9BQU8sQ0FBQztBQUM3QixPQUFPLEVBQWdCLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNuRCxPQUFPLEVBQUMsc0JBQXNCLEVBQUUsb0JBQW9CLEVBQVcsTUFBTSxRQUFRLENBQUM7QUFDOUUsT0FBTyxFQUE0QixnQkFBZ0IsRUFBRSxNQUFNLEVBQWlCLFFBQVEsRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUd0RyxPQUFPLEVBQUMscUJBQXFCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFcEQsT0FBTyxLQUFLLElBQUksTUFBTSxRQUFRLENBQUM7QUFDL0IsT0FBTyxFQUFDLG9CQUFvQixFQUFFLGtCQUFrQixFQUFFLEdBQUcsRUFBRSxhQUFhLEVBQUMsTUFBTSxRQUFRLENBQUM7QUF1RXBGLFNBQVMsNEJBQTRCLENBRWpDLGdCQUNnQztJQUVsQyxPQUFRLGdCQUFrRCxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUM7QUFDaEYsQ0FBQztBQUVELE1BQU0sV0FBVztJQUFqQjtRQUNFLHVDQUF1QztRQUN2Qyx3QkFBbUIs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SztTQXRVdkIsTUFBTTtBQTRsQ25CLFNBQVMsSUFBSSxDQUFDLEtBQWU7SUFDM0IsTUFBTSxNQUFNLEdBQUcsa0JBQWtCLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQ25FLE9BQU8sTUFBTSxDQUFDLFVBQVUsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0FBQ3JELENBQUM7QUFFRCxNQUFNLFVBQVUsZUFBZTtJQUM3QixNQUFNLEVBQUUsR0FBRyxrQkFBa0IsRUFBb0MsQ0FBQztJQUNsRSxJQUFJLEVBQUUsQ0FBQyxTQUFTLElBQUksSUFBSSxFQUFFO1FBQ3hCLE1BQU0sV0FBVyxHQUFHLElBQUksV0FBVyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQ3hDLEVBQUUsQ0FBQyxTQUFTLEdBQUcsSUFBSSxNQUFNLENBQUMsV0FBVyxDQUFDLENBQUM7S0FDeEM7SUFDRCxvQkFBb0IsQ0FBQyxFQUFFLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBRXZDLDBFQUEwRTtJQUMxRSxnQkFBZ0I7SUFDaEIsZ0JBQWdCLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQ3JDLE9BQU8sRUFBRSxDQUFDLFNBQVMsQ0FBQztBQUN0QixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sTUFBTSxHQUFHLGVBQWUsRUFBRSxDQUFDO0FBRXhDOzs7OztHQUtHO0FBQ0gsTUFBTSxVQUFVLEdBQUcsQ0FBQyxDQUFTLEVBQUUsQ0FBUztJQUN0QyxvRUFBb0U7SUFDcEUsTUFBTSxNQUFNLEdBQUcsRUFBQyxDQUFDLEVBQUUsQ0FBQyxFQUFDLENBQUM7SUFDdEIsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLEdBQUcsRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDcEUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtCYWNrZW5kVGltaW5nSW5mbywgRGF0YU1vdmVyLCBLZXJuZWxCYWNrZW5kfSBmcm9tICcuL2JhY2tlbmRzL2JhY2tlbmQnO1xuaW1wb3J0IHtFbnZpcm9ubWVudCwgc2V0RW52aXJvbm1lbnRHbG9iYWx9IGZyb20gJy4vZW52aXJvbm1lbnQnO1xuaW1wb3J0IHtnZXRHbG9iYWxOYW1lc3BhY2V9IGZyb20gJy4vZ2xvYmFsX3V0aWwnO1xuaW1wb3J0IHtBZGQsIENhc3QsIElkZW50aXR5fSBmcm9tICcuL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQgeyBnZXRHcmFkaWVudCwgZ2V0S2VybmVsLCBnZXRLZXJuZWxzRm9yQmFja2VuZCwgR3JhZEZ1bmMsIE5hbWVkQXR0ck1hcCB9IGZyb20gJy4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7IFRlbnNvckluZm8gfSBmcm9tICcuL3RlbnNvcl9pbmZvJztcbmltcG9ydCAqIGFzIGxvZyBmcm9tICcuL2xvZyc7XG5pbXBvcnQge0tlcm5lbFByb2ZpbGUsIFByb2ZpbGVyfSBmcm9tICcuL3Byb2ZpbGVyJztcbmltcG9ydCB7YmFja3Byb3BhZ2F0ZUdyYWRpZW50cywgZ2V0RmlsdGVyZWROb2Rlc1hUb1ksIFRhcGVOb2RlfSBmcm9tICcuL3RhcGUnO1xuaW1wb3J0IHtEYXRhVG9HUFVPcHRpb25zLCBHUFVEYXRhLCBzZXRUZW5zb3JUcmFja2VyLCBUZW5zb3IsIFRlbnNvclRyYWNrZXIsIFZhcmlhYmxlfSBmcm9tICcuL3RlbnNvcic7XG5pbXBvcnQge0RhdGFJZH0gZnJvbSAnLi90ZW5zb3JfaW5mbyc7XG5pbXBvcnQge0dyYWRTYXZlRnVuYywgTmFtZWRUZW5zb3JNYXAsIE5hbWVkVmFyaWFibGVNYXAsIFRlbnNvckNvbnRhaW5lcn0gZnJvbSAnLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtnZXRUZW5zb3JzSW5Db250YWluZXJ9IGZyb20gJy4vdGVuc29yX3V0aWwnO1xuaW1wb3J0IHtCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZSwgRGF0YVZhbHVlc30gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4vdXRpbCc7XG5pbXBvcnQge2J5dGVzRnJvbVN0cmluZ0FycmF5LCBtYWtlT25lc1R5cGVkQXJyYXksIG5vdywgc2l6ZUZyb21TaGFwZX0gZnJvbSAnLi91dGlsJztcblxuLyoqXG4gKiBBIGZ1bmN0aW9uIHRoYXQgY29tcHV0ZXMgYW4gb3V0cHV0LiBUaGUgc2F2ZSBmdW5jdGlvbiBpcyBmb3Igc2F2aW5nIHRlbnNvcnNcbiAqIGNvbXB1dGVkIGluIHRoZSBmb3J3YXJkIHBhc3MsIHRoYXQgd2UgbmVlZCBpbiB0aGUgYmFja3dhcmQgcGFzcy5cbiAqL1xuZXhwb3J0IHR5cGUgRm9yd2FyZEZ1bmM8VD4gPSAoYmFja2VuZDogS2VybmVsQmFja2VuZCwgc2F2ZT86IEdyYWRTYXZlRnVuYykgPT4gVDtcblxuLyoqXG4gKiBAZG9jYWxpYXMgKGE6IFRlbnNvciwgYjogVGVuc29yLC4uLiwgc2F2ZT86IEZ1bmN0aW9uKSA9PiB7XG4gKiAgIHZhbHVlOiBUZW5zb3IsXG4gKiAgIGdyYWRGdW5jOiAoZHk6IFRlbnNvciwgc2F2ZWQ/OiBOYW1lZFRlbnNvck1hcCkgPT4gVGVuc29yIHwgVGVuc29yW11cbiAqIH1cbiAqL1xuZXhwb3J0IHR5cGUgQ3VzdG9tR3JhZGllbnRGdW5jPFQgZXh0ZW5kcyBUZW5zb3I+ID1cbiAgICAoLi4uaW5wdXRzOiBBcnJheTxUZW5zb3J8R3JhZFNhdmVGdW5jPikgPT4ge1xuICAgICAgdmFsdWU6IFQ7XG4gICAgICBncmFkRnVuYzogKGR5OiBULCBzYXZlZDogVGVuc29yW10pID0+IFRlbnNvciB8IFRlbnNvcltdO1xuICAgIH07XG5cbmV4cG9ydCB0eXBlIE1lbW9yeUluZm8gPSB7XG4gIG51bVRlbnNvcnM6IG51bWJlcjsgbnVtRGF0YUJ1ZmZlcnM6IG51bWJlcjsgbnVtQnl0ZXM6IG51bWJlcjtcbiAgdW5yZWxpYWJsZT86IGJvb2xlYW47IHJlYXNvbnM6IHN0cmluZ1tdO1xufTtcblxudHlwZSBLZXJuZWxJbmZvID0ge1xuICBuYW1lOiBzdHJpbmc7IGJ5dGVzQWRkZWQ6IG51bWJlcjsgdG90YWxCeXRlc1NuYXBzaG90OiBudW1iZXI7XG4gIHRlbnNvcnNBZGRlZDogbnVtYmVyO1xuICB0b3RhbFRlbnNvcnNTbmFwc2hvdDogbnVtYmVyO1xuICBpbnB1dFNoYXBlczogbnVtYmVyW11bXTtcbiAgb3V0cHV0U2hhcGVzOiBudW1iZXJbXVtdO1xuICBrZXJuZWxUaW1lTXM6IG51bWJlciB8IHtlcnJvcjogc3RyaW5nfSB8IFByb21pc2U8bnVtYmVyfHtlcnJvcjogc3RyaW5nfT47XG4gIGV4dHJhSW5mbzogc3RyaW5nIHwgUHJvbWlzZTxzdHJpbmc+O1xufTtcblxuZXhwb3J0IHR5cGUgUHJvZmlsZUluZm8gPSB7XG4gIG5ld0J5dGVzOiBudW1iZXI7IG5ld1RlbnNvcnM6IG51bWJlcjsgcGVha0J5dGVzOiBudW1iZXI7XG4gIGtlcm5lbHM6IEtlcm5lbEluZm9bXTtcbiAgcmVzdWx0OiBUZW5zb3JDb250YWluZXI7XG4gIGtlcm5lbE5hbWVzOiBzdHJpbmdbXTtcbn07XG5cbmV4cG9ydCBpbnRlcmZhY2UgVGltaW5nSW5mbyBleHRlbmRzIEJhY2tlbmRUaW1pbmdJbmZvIHtcbiAgd2FsbE1zOiBudW1iZXI7XG59XG5cbi8qKiBAZG9jYWxpYXMgRnVuY3Rpb24gKi9cbmV4cG9ydCB0eXBlIFNjb3BlRm48VCBleHRlbmRzIFRlbnNvckNvbnRhaW5lcj4gPSAoKSA9PiBUO1xuXG5pbnRlcmZhY2UgU2NvcGVTdGF0ZSB7XG4gIHRyYWNrOiBUZW5zb3JbXTtcbiAgbmFtZTogc3RyaW5nO1xuICBpZDogbnVtYmVyO1xufVxuXG5pbnRlcmZhY2UgUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb248SSBleHRlbmRzIE5hbWVkVGVuc29yTWFwPiB7XG4gIGtlcm5lbE5hbWU6IHN0cmluZztcbiAgaW5wdXRzOiBJO1xuICBhdHRycz86IE5hbWVkQXR0ck1hcDtcbn1cblxuaW50ZXJmYWNlIEN1c3RvbUdyYWRLZXJuZWxJbnZvY2F0aW9uPFQgZXh0ZW5kcyBUZW5zb3J8VGVuc29yW10sXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEkgZXh0ZW5kcyBOYW1lZFRlbnNvck1hcD4ge1xuICBmb3J3YXJkRnVuYzogRm9yd2FyZEZ1bmM8VD47XG4gIGJhY2t3YXJkc0Z1bmM6IChkeTogVCwgc2F2ZWQ6IFRlbnNvcltdKSA9PiB7XG4gICAgW1AgaW4ga2V5b2YgSV06ICgpID0+IElbUF1cbiAgfTtcbiAgaW5wdXRzOiBJO1xuICBhdHRycz86IE5hbWVkQXR0ck1hcDtcbn1cblxuZnVuY3Rpb24gaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxUIGV4dGVuZHMgVGVuc29yfFRlbnNvcltdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgSSBleHRlbmRzIE5hbWVkVGVuc29yTWFwPihcbiAgICBrZXJuZWxJbnZvY2F0aW9uOiBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJPnxcbiAgICBDdXN0b21HcmFkS2VybmVsSW52b2NhdGlvbjxULCBJPik6XG4gICAga2VybmVsSW52b2NhdGlvbiBpcyBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJPiB7XG4gIHJldHVybiAoa2VybmVsSW52b2NhdGlvbiBhcyBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJPikua2VybmVsTmFtZSAhPSBudWxsO1xufVxuXG5jbGFzcyBFbmdpbmVTdGF0ZSB7XG4gIC8vIFB1YmxpYyBzaW5jZSBvcHRpbWl6ZXJzIHdpbGwgdXNlIGl0LlxuICByZWdpc3RlcmVkVmFyaWFibGVzOiBOYW1lZFZhcmlhYmxlTWFwID0ge307XG5cbiAgbmV4dFRhcGVOb2RlSWQgPSAwO1xuICBudW1CeXRlcyA9IDA7XG4gIG51bVRlbnNvcnMgPSAwO1xuICBudW1TdHJpbmdUZW5zb3JzID0gMDtcbiAgbnVtRGF0YUJ1ZmZlcnMgPSAwO1xuXG4gIGFjdGl2ZVRhcGU6IFRhcGVOb2RlW107XG4gIC8vIE51bWJlciBvZiBuZXN0ZWQgdGYuZ3JhZCgpIHN0YXRlbWVudHMgd2hlbiBjb21wdXRpbmcgaGlnaGVyLW9yZGVyXG4gIC8vIGdyYWRpZW50cy4gRS5nLiBgMWAgZm9yIGZpcnN0LW9yZGVyIGdyYWRpZW50cyBhbmQgYDJgIGZvciBzZWNvbmQtb3JkZXJcbiAgLy8gZ3JhZGllbnRzLiBVc2VkIHRvIHRyYWNrIGlmIHRoZSB0YXBlIHNob3VsZCBiZSByZW1vdmVkIGFmdGVyIGEgYmFja3Byb3AuXG4gIGdyYWRpZW50RGVwdGggPSAwO1xuICAvLyBOdW1iZXIgb2YgbmVzdGVkIGtlcm5lbCBjYWxscy4gV2hlbiBrZXJuZWwgZGVwdGggaXMgZ3JlYXRlciB0aGFuIDEsIHdlIHR1cm5cbiAgLy8gb2ZmIHRoZSB0YXBlLlxuICBrZXJuZWxEZXB0aCA9IDA7XG5cbiAgLy8gS2VlcCBUZW5zb3JzIHRoYXQgcGFyYWxsZWwgdGhlIHRhcGVzLlxuICBhY3RpdmVTY29wZTogU2NvcGVTdGF0ZTtcbiAgc2NvcGVTdGFjazogU2NvcGVTdGF0ZVtdID0gW107XG4gIC8qKlxuICAgKiBLZWVwcyB0cmFjayBvZiB0aGUgbnVtYmVyIG9mIGRhdGEgbW92ZXMgZHVyaW5nIGEga2VybmVsIGV4ZWN1dGlvbi4gV2VcbiAgICogbWFpbnRhaW4gYSBzdGFjayBzaW5jZSBrZXJuZWxzIGNhbiBjYWxsIG90aGVyIGtlcm5lbHMsIHJlY3Vyc2l2ZWx5LlxuICAgKi9cbiAgbnVtRGF0YU1vdmVzU3RhY2s6IG51bWJlcltdID0gW107XG4gIG5leHRTY29wZUlkID0gMDtcblxuICB0ZW5zb3JJbmZvID0gbmV3IFdlYWtNYXA8RGF0YUlkLCB7XG4gICAgYmFja2VuZDogS2VybmVsQmFja2VuZCxcbiAgICBieXRlczogbnVtYmVyLFxuICAgIGR0eXBlOiBEYXRhVHlwZSxcbiAgICBzaGFwZTogbnVtYmVyW11cbiAgfT4oKTtcblxuICBwcm9maWxpbmcgPSBmYWxzZTtcbiAgYWN0aXZlUHJvZmlsZTogUHJvZmlsZUluZm8gPSB7XG4gICAgbmV3Qnl0ZXM6IDAsXG4gICAgbmV3VGVuc29yczogMCxcbiAgICBwZWFrQnl0ZXM6IDAsXG4gICAga2VybmVsczogW10sXG4gICAgcmVzdWx0OiBudWxsLFxuICAgIGdldCBrZXJuZWxOYW1lcygpOlxuICAgICAgICBzdHJpbmdbXSB7XG4gICAgICAgICAgcmV0dXJuIEFycmF5LmZyb20obmV3IFNldCh0aGlzLmtlcm5lbHMubWFwKGsgPT4gay5uYW1lKSkpO1xuICAgICAgICB9XG4gIH07XG5cbiAgZGlzcG9zZSgpIHtcbiAgICBmb3IgKGNvbnN0IHZhcmlhYmxlTmFtZSBpbiB0aGlzLnJlZ2lzdGVyZWRWYXJpYWJsZXMpIHtcbiAgICAgIHRoaXMucmVnaXN0ZXJlZFZhcmlhYmxlc1t2YXJpYWJsZU5hbWVdLmRpc3Bvc2UoKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEVuZ2luZSBpbXBsZW1lbnRzIFRlbnNvclRyYWNrZXIsIERhdGFNb3ZlciB7XG4gIHN0YXRlOiBFbmdpbmVTdGF0ZTtcbiAgYmFja2VuZE5hbWU6IHN0cmluZztcbiAgcmVnaXN0cnk6IHtbaWQ6IHN0cmluZ106IEtlcm5lbEJhY2tlbmR9ID0ge307XG4gIHJlZ2lzdHJ5RmFjdG9yeToge1xuICAgIFtpZDogc3RyaW5nXToge1xuICAgICAgZmFjdG9yeTogKCkgPT4gS2VybmVsQmFja2VuZCB8IFByb21pc2U8S2VybmVsQmFja2VuZD4sXG4gICAgICBwcmlvcml0eTogbnVtYmVyXG4gICAgfVxuICB9ID0ge307XG5cbiAgcHJpdmF0ZSBwcm9maWxlcjogUHJvZmlsZXI7XG4gIHByaXZhdGUgYmFja2VuZEluc3RhbmNlOiBLZXJuZWxCYWNrZW5kO1xuICBwcml2YXRlIHBlbmRpbmdCYWNrZW5kSW5pdDogUHJvbWlzZTxib29sZWFuPjtcbiAgcHJpdmF0ZSBwZW5kaW5nQmFja2VuZEluaXRJZCA9IDA7XG5cbiAgY29uc3RydWN0b3IocHVibGljIEVOVjogRW52aXJvbm1lbnQpIHtcbiAgICB0aGlzLnN0YXRlID0gbmV3IEVuZ2luZVN0YXRlKCk7XG4gIH1cblxuICBhc3luYyByZWFkeSgpOiBQcm9taXNlPHZvaWQ+IHtcbiAgICBpZiAodGhpcy5wZW5kaW5nQmFja2VuZEluaXQgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0LnRoZW4oKCkgPT4ge30pO1xuICAgIH1cbiAgICBpZiAodGhpcy5iYWNrZW5kSW5zdGFuY2UgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBzb3J0ZWRCYWNrZW5kcyA9IHRoaXMuZ2V0U29ydGVkQmFja2VuZHMoKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgc29ydGVkQmFja2VuZHMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IGJhY2tlbmROYW1lID0gc29ydGVkQmFja2VuZHNbaV07XG4gICAgICBjb25zdCBzdWNjZXNzID0gYXdhaXQgdGhpcy5pbml0aWFsaXplQmFja2VuZChiYWNrZW5kTmFtZSkuc3VjY2VzcztcbiAgICAgIGlmIChzdWNjZXNzKSB7XG4gICAgICAgIGF3YWl0IHRoaXMuc2V0QmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBDb3VsZCBub3QgaW5pdGlhbGl6ZSBhbnkgYmFja2VuZHMsIGFsbCBiYWNrZW5kIGluaXRpYWxpemF0aW9ucyBgICtcbiAgICAgICAgYGZhaWxlZC5gKTtcbiAgfVxuXG4gIGdldCBiYWNrZW5kKCk6IEtlcm5lbEJhY2tlbmQge1xuICAgIGlmICh0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEJhY2tlbmQgJyR7dGhpcy5iYWNrZW5kTmFtZX0nIGhhcyBub3QgeWV0IGJlZW4gaW5pdGlhbGl6ZWQuIE1ha2UgYCArXG4gICAgICAgICAgYHN1cmUgdG8gYXdhaXQgdGYucmVhZHkoKSBvciBhd2FpdCB0Zi5zZXRCYWNrZW5kKCkgYmVmb3JlIGNhbGxpbmcgYCArXG4gICAgICAgICAgYG90aGVyIG1ldGhvZHNgKTtcbiAgICB9XG4gICAgaWYgKHRoaXMuYmFja2VuZEluc3RhbmNlID09IG51bGwpIHtcbiAgICAgIGNvbnN0IHtuYW1lLCBhc3luY0luaXR9ID0gdGhpcy5pbml0aWFsaXplQmFja2VuZHNBbmRSZXR1cm5CZXN0KCk7XG4gICAgICBpZiAoYXN5bmNJbml0KSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBUaGUgaGlnaGVzdCBwcmlvcml0eSBiYWNrZW5kICcke25hbWV9JyBoYXMgbm90IHlldCBiZWVuIGAgK1xuICAgICAgICAgICAgYGluaXRpYWxpemVkLiBNYWtlIHN1cmUgdG8gYXdhaXQgdGYucmVhZHkoKSBvciBgICtcbiAgICAgICAgICAgIGBhd2FpdCB0Zi5zZXRCYWNrZW5kKCkgYmVmb3JlIGNhbGxpbmcgb3RoZXIgbWV0aG9kc2ApO1xuICAgICAgfVxuICAgICAgdGhpcy5zZXRCYWNrZW5kKG5hbWUpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5iYWNrZW5kSW5zdGFuY2U7XG4gIH1cblxuICBiYWNrZW5kTmFtZXMoKTogc3RyaW5nW10ge1xuICAgIHJldHVybiBPYmplY3Qua2V5cyh0aGlzLnJlZ2lzdHJ5RmFjdG9yeSk7XG4gIH1cblxuICBmaW5kQmFja2VuZChiYWNrZW5kTmFtZTogc3RyaW5nKTogS2VybmVsQmFja2VuZCB7XG4gICAgaWYgKCEoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeSkpIHtcbiAgICAgIC8vIElmIHRoZSBiYWNrZW5kIGhhc24ndCBiZWVuIGluaXRpYWxpemVkIGJ1dCB3ZSBoYXZlIGEgcmVnaXN0cnkgZW50cnkgZm9yXG4gICAgICAvLyBpdCwgaW5pdGlhbGl6ZSBpdCBhbmQgcmV0dXJuIGl0LlxuICAgICAgaWYgKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnlGYWN0b3J5KSB7XG4gICAgICAgIGNvbnN0IHthc3luY0luaXR9ID0gdGhpcy5pbml0aWFsaXplQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgICAgIGlmIChhc3luY0luaXQpIHtcbiAgICAgICAgICAvLyBCYWNrZW5kIGlzIG5vdCByZWFkeSB5ZXQuXG4gICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBudWxsO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV07XG4gIH1cblxuICBmaW5kQmFja2VuZEZhY3RvcnkoYmFja2VuZE5hbWU6IHN0cmluZyk6XG4gICAgICAoKSA9PiBLZXJuZWxCYWNrZW5kIHwgUHJvbWlzZTxLZXJuZWxCYWNrZW5kPiB7XG4gICAgaWYgKCEoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeUZhY3RvcnkpKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMucmVnaXN0cnlGYWN0b3J5W2JhY2tlbmROYW1lXS5mYWN0b3J5O1xuICB9XG5cbiAgcmVnaXN0ZXJCYWNrZW5kKFxuICAgICAgYmFja2VuZE5hbWU6IHN0cmluZyxcbiAgICAgIGZhY3Rvcnk6ICgpID0+IEtlcm5lbEJhY2tlbmQgfCBQcm9taXNlPEtlcm5lbEJhY2tlbmQ+LFxuICAgICAgcHJpb3JpdHkgPSAxKTogYm9vbGVhbiB7XG4gICAgaWYgKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnlGYWN0b3J5KSB7XG4gICAgICBsb2cud2FybihcbiAgICAgICAgICBgJHtiYWNrZW5kTmFtZX0gYmFja2VuZCB3YXMgYWxyZWFkeSByZWdpc3RlcmVkLiBgICtcbiAgICAgICAgICBgUmV1c2luZyBleGlzdGluZyBiYWNrZW5kIGZhY3RvcnkuYCk7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIHRoaXMucmVnaXN0cnlGYWN0b3J5W2JhY2tlbmROYW1lXSA9IHtmYWN0b3J5LCBwcmlvcml0eX07XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICBhc3luYyBzZXRCYWNrZW5kKGJhY2tlbmROYW1lOiBzdHJpbmcpOiBQcm9taXNlPGJvb2xlYW4+IHtcbiAgICBpZiAodGhpcy5yZWdpc3RyeUZhY3RvcnlbYmFja2VuZE5hbWVdID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgQmFja2VuZCBuYW1lICcke2JhY2tlbmROYW1lfScgbm90IGZvdW5kIGluIHJlZ2lzdHJ5YCk7XG4gICAgfVxuICAgIHRoaXMuYmFja2VuZE5hbWUgPSBiYWNrZW5kTmFtZTtcbiAgICBpZiAodGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0gPT0gbnVsbCkge1xuICAgICAgdGhpcy5iYWNrZW5kSW5zdGFuY2UgPSBudWxsO1xuICAgICAgY29uc3Qge3N1Y2Nlc3MsIGFzeW5jSW5pdH0gPSB0aGlzLmluaXRpYWxpemVCYWNrZW5kKGJhY2tlbmROYW1lKTtcbiAgICAgIGNvbnN0IHJlc3VsdCA9IGFzeW5jSW5pdCA/IGF3YWl0IHN1Y2Nlc3MgOiBzdWNjZXNzO1xuICAgICAgaWYgKCFyZXN1bHQpIHtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cbiAgICB0aGlzLmJhY2tlbmRJbnN0YW5jZSA9IHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdO1xuICAgIHRoaXMuc2V0dXBSZWdpc3RlcmVkS2VybmVscygpO1xuICAgIC8vIFJlc2V0IHRoZSBwcm9maWxlci5cbiAgICB0aGlzLnByb2ZpbGVyID0gbmV3IFByb2ZpbGVyKHRoaXMuYmFja2VuZEluc3RhbmNlKTtcblxuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgcHJpdmF0ZSBzZXR1cFJlZ2lzdGVyZWRLZXJuZWxzKCk6IHZvaWQge1xuICAgIGNvbnN0IGtlcm5lbHMgPSBnZXRLZXJuZWxzRm9yQmFja2VuZCh0aGlzLmJhY2tlbmROYW1lKTtcbiAgICBrZXJuZWxzLmZvckVhY2goa2VybmVsID0+IHtcbiAgICAgIGlmIChrZXJuZWwuc2V0dXBGdW5jICE9IG51bGwpIHtcbiAgICAgICAga2VybmVsLnNldHVwRnVuYyh0aGlzLmJhY2tlbmRJbnN0YW5jZSk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBwcml2YXRlIGRpc3Bvc2VSZWdpc3RlcmVkS2VybmVscyhiYWNrZW5kTmFtZTogc3RyaW5nKTogdm9pZCB7XG4gICAgY29uc3Qga2VybmVscyA9IGdldEtlcm5lbHNGb3JCYWNrZW5kKGJhY2tlbmROYW1lKTtcbiAgICBrZXJuZWxzLmZvckVhY2goa2VybmVsID0+IHtcbiAgICAgIGlmIChrZXJuZWwuZGlzcG9zZUZ1bmMgIT0gbnVsbCkge1xuICAgICAgICBrZXJuZWwuZGlzcG9zZUZ1bmModGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0pO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVzIGEgYmFja2VuZCBieSBsb29raW5nIHVwIHRoZSBiYWNrZW5kIG5hbWUgaW4gdGhlIGZhY3RvcnlcbiAgICogcmVnaXN0cnkgYW5kIGNhbGxpbmcgdGhlIGZhY3RvcnkgbWV0aG9kLiBSZXR1cm5zIGEgYm9vbGVhbiByZXByZXNlbnRpbmdcbiAgICogd2hldGhlciB0aGUgaW5pdGlhbGl6YXRpb24gb2YgdGhlIGJhY2tlbmQgc3VjZWVkZWQuIFRocm93cyBhbiBlcnJvciBpZlxuICAgKiB0aGVyZSBpcyBubyBiYWNrZW5kIGluIHRoZSBmYWN0b3J5IHJlZ2lzdHJ5LlxuICAgKi9cbiAgcHJpdmF0ZSBpbml0aWFsaXplQmFja2VuZChiYWNrZW5kTmFtZTogc3RyaW5nKTpcbiAgICAgIHtzdWNjZXNzOiBib29sZWFufFByb21pc2U8Ym9vbGVhbj4sIGFzeW5jSW5pdDogYm9vbGVhbn0ge1xuICAgIGNvbnN0IHJlZ2lzdHJ5RmFjdG9yeUVudHJ5ID0gdGhpcy5yZWdpc3RyeUZhY3RvcnlbYmFja2VuZE5hbWVdO1xuICAgIGlmIChyZWdpc3RyeUZhY3RvcnlFbnRyeSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCBpbml0aWFsaXplIGJhY2tlbmQgJHtiYWNrZW5kTmFtZX0sIG5vIHJlZ2lzdHJhdGlvbiBmb3VuZC5gKTtcbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgY29uc3QgYmFja2VuZCA9IHJlZ2lzdHJ5RmFjdG9yeUVudHJ5LmZhY3RvcnkoKTtcbiAgICAgIC8qIFRlc3QgaWYgdGhlIGZhY3RvcnkgcmV0dXJucyBhIHByb21pc2UuXG4gICAgICBEb25lIGluIGEgbW9yZSBsaWJlcmFsIHdheSB0aGFuXG4gICAgICBwcmV2aW91cyAnUHJvbWlzZS5yZXNvbHZlKGJhY2tlbmQpPT09YmFja2VuZCdcbiAgICAgIGFzIHdlIG5lZWRlZCB0byBhY2NvdW50IGZvciBjdXN0b20gUHJvbWlzZVxuICAgICAgaW1wbGVtZW50YXRpb25zIChlLmcuIEFuZ3VsYXIpICovXG4gICAgICBpZiAoYmFja2VuZCAmJiAhKGJhY2tlbmQgaW5zdGFuY2VvZiBLZXJuZWxCYWNrZW5kKSAmJlxuICAgICAgICAgIHR5cGVvZiBiYWNrZW5kLnRoZW4gPT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgY29uc3QgcHJvbWlzZUlkID0gKyt0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdElkO1xuICAgICAgICBjb25zdCBzdWNjZXNzID1cbiAgICAgICAgICAgIGJhY2tlbmRcbiAgICAgICAgICAgICAgICAudGhlbihiYWNrZW5kSW5zdGFuY2UgPT4ge1xuICAgICAgICAgICAgICAgICAgLy8gT3V0ZGF0ZWQgcHJvbWlzZS4gQW5vdGhlciBiYWNrZW5kIHdhcyBzZXQgaW4gdGhlIG1lYW50aW1lLlxuICAgICAgICAgICAgICAgICAgaWYgKHByb21pc2VJZCA8IHRoaXMucGVuZGluZ0JhY2tlbmRJbml0SWQpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0gPSBiYWNrZW5kSW5zdGFuY2U7XG4gICAgICAgICAgICAgICAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCA9IG51bGw7XG4gICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9KVxuICAgICAgICAgICAgICAgIC5jYXRjaChlcnIgPT4ge1xuICAgICAgICAgICAgICAgICAgLy8gT3V0ZGF0ZWQgcHJvbWlzZS4gQW5vdGhlciBiYWNrZW5kIHdhcyBzZXQgaW4gdGhlIG1lYW50aW1lLlxuICAgICAgICAgICAgICAgICAgaWYgKHByb21pc2VJZCA8IHRoaXMucGVuZGluZ0JhY2tlbmRJbml0SWQpIHtcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgPSBudWxsO1xuICAgICAgICAgICAgICAgICAgbG9nLndhcm4oYEluaXRpYWxpemF0aW9uIG9mIGJhY2tlbmQgJHtiYWNrZW5kTmFtZX0gZmFpbGVkYCk7XG4gICAgICAgICAgICAgICAgICBsb2cud2FybihlcnIuc3RhY2sgfHwgZXJyLm1lc3NhZ2UpO1xuICAgICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCA9IHN1Y2Nlc3M7XG4gICAgICAgIHJldHVybiB7c3VjY2VzcywgYXN5bmNJbml0OiB0cnVlfTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdID0gYmFja2VuZCBhcyBLZXJuZWxCYWNrZW5kO1xuICAgICAgICByZXR1cm4ge3N1Y2Nlc3M6IHRydWUsIGFzeW5jSW5pdDogZmFsc2V9O1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgbG9nLndhcm4oYEluaXRpYWxpemF0aW9uIG9mIGJhY2tlbmQgJHtiYWNrZW5kTmFtZX0gZmFpbGVkYCk7XG4gICAgICBsb2cud2FybihlcnIuc3RhY2sgfHwgZXJyLm1lc3NhZ2UpO1xuICAgICAgcmV0dXJuIHtzdWNjZXNzOiBmYWxzZSwgYXN5bmNJbml0OiBmYWxzZX07XG4gICAgfVxuICB9XG5cbiAgcmVtb3ZlQmFja2VuZChiYWNrZW5kTmFtZTogc3RyaW5nKTogdm9pZCB7XG4gICAgaWYgKCEoYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeUZhY3RvcnkpKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYCR7YmFja2VuZE5hbWV9IGJhY2tlbmQgbm90IGZvdW5kIGluIHJlZ2lzdHJ5YCk7XG4gICAgfVxuICAgIGlmICh0aGlzLmJhY2tlbmROYW1lID09PSBiYWNrZW5kTmFtZSAmJiB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCAhPSBudWxsKSB7XG4gICAgICAvLyBUaGVyZSBpcyBhIHBlbmRpbmcgcHJvbWlzZSBvZiB0aGUgYmFja2VuZCB3ZSB3YW50IHRvIHJlbW92ZS4gTWFrZSBpdFxuICAgICAgLy8gb2Jzb2xldGUuXG4gICAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdElkKys7XG4gICAgfVxuXG4gICAgaWYgKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnkpIHtcbiAgICAgIHRoaXMuZGlzcG9zZVJlZ2lzdGVyZWRLZXJuZWxzKGJhY2tlbmROYW1lKTtcbiAgICAgIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdLmRpc3Bvc2UoKTtcbiAgICAgIGRlbGV0ZSB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXTtcbiAgICB9XG5cbiAgICBkZWxldGUgdGhpcy5yZWdpc3RyeUZhY3RvcnlbYmFja2VuZE5hbWVdO1xuXG4gICAgLy8gVW5zZXQgdGhlIGJhY2tlbmQgaWYgaXQgaXMgYWN0aXZlLlxuICAgIGlmICh0aGlzLmJhY2tlbmROYW1lID09PSBiYWNrZW5kTmFtZSkge1xuICAgICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgPSBudWxsO1xuICAgICAgdGhpcy5iYWNrZW5kTmFtZSA9IG51bGw7XG4gICAgICB0aGlzLmJhY2tlbmRJbnN0YW5jZSA9IG51bGw7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRTb3J0ZWRCYWNrZW5kcygpOiBzdHJpbmdbXSB7XG4gICAgaWYgKE9iamVjdC5rZXlzKHRoaXMucmVnaXN0cnlGYWN0b3J5KS5sZW5ndGggPT09IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTm8gYmFja2VuZCBmb3VuZCBpbiByZWdpc3RyeS4nKTtcbiAgICB9XG4gICAgcmV0dXJuIE9iamVjdC5rZXlzKHRoaXMucmVnaXN0cnlGYWN0b3J5KS5zb3J0KChhOiBzdHJpbmcsIGI6IHN0cmluZykgPT4ge1xuICAgICAgLy8gSGlnaGVzdCBwcmlvcml0eSBjb21lcyBmaXJzdC5cbiAgICAgIHJldHVybiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiXS5wcmlvcml0eSAtXG4gICAgICAgICAgdGhpcy5yZWdpc3RyeUZhY3RvcnlbYV0ucHJpb3JpdHk7XG4gICAgfSk7XG4gIH1cblxuICBwcml2YXRlIGluaXRpYWxpemVCYWNrZW5kc0FuZFJldHVybkJlc3QoKTpcbiAgICAgIHtuYW1lOiBzdHJpbmcsIGFzeW5jSW5pdDogYm9vbGVhbn0ge1xuICAgIGNvbnN0IHNvcnRlZEJhY2tlbmRzID0gdGhpcy5nZXRTb3J0ZWRCYWNrZW5kcygpO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzb3J0ZWRCYWNrZW5kcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgYmFja2VuZE5hbWUgPSBzb3J0ZWRCYWNrZW5kc1tpXTtcbiAgICAgIGNvbnN0IHtzdWNjZXNzLCBhc3luY0luaXR9ID0gdGhpcy5pbml0aWFsaXplQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgICBpZiAoYXN5bmNJbml0IHx8IHN1Y2Nlc3MpIHtcbiAgICAgICAgcmV0dXJuIHtuYW1lOiBiYWNrZW5kTmFtZSwgYXN5bmNJbml0fTtcbiAgICAgIH1cbiAgICB9XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgQ291bGQgbm90IGluaXRpYWxpemUgYW55IGJhY2tlbmRzLCBhbGwgYmFja2VuZCBpbml0aWFsaXphdGlvbnMgYCArXG4gICAgICAgIGBmYWlsZWQuYCk7XG4gIH1cblxuICBtb3ZlRGF0YShiYWNrZW5kOiBLZXJuZWxCYWNrZW5kLCBkYXRhSWQ6IERhdGFJZCkge1xuICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qgc3JjQmFja2VuZCA9IGluZm8uYmFja2VuZDtcbiAgICBjb25zdCB2YWx1ZXMgPSB0aGlzLnJlYWRTeW5jKGRhdGFJZCk7XG4gICAgY29uc3QgcmVmQ291bnQgPSBzcmNCYWNrZW5kLnJlZkNvdW50KGRhdGFJZCk7XG4gICAgLy8gRGVsZXRlIHRoZSB0ZW5zb3IgZnJvbSB0aGUgb2xkIGJhY2tlbmQgYW5kIG1vdmUgaXQgdG8gdGhlIG5ld1xuICAgIC8vIGJhY2tlbmQuXG4gICAgc3JjQmFja2VuZC5kaXNwb3NlRGF0YShkYXRhSWQsIHRydWUpO1xuICAgIGluZm8uYmFja2VuZCA9IGJhY2tlbmQ7XG4gICAgYmFja2VuZC5tb3ZlKGRhdGFJZCwgdmFsdWVzLCBpbmZvLnNoYXBlLCBpbmZvLmR0eXBlLCByZWZDb3VudCk7XG4gICAgaWYgKHRoaXMuc2hvdWxkQ2hlY2tGb3JNZW1MZWFrcygpKSB7XG4gICAgICAvLyBUcmFjayB0aGUgbnVtYmVyIG9mIG1vdmVzIGR1cmluZyBhIGtlcm5lbCBleGVjdXRpb24gdG8gY29ycmVjdGx5XG4gICAgICAvLyBkZXRlY3QgbWVtb3J5IGxlYWtzLlxuICAgICAgdGhpcy5zdGF0ZS5udW1EYXRhTW92ZXNTdGFja1t0aGlzLnN0YXRlLm51bURhdGFNb3Zlc1N0YWNrLmxlbmd0aCAtIDFdKys7XG4gICAgfVxuICB9XG5cbiAgdGlkeTxUIGV4dGVuZHMgVGVuc29yQ29udGFpbmVyPihuYW1lT3JGbjogc3RyaW5nfFNjb3BlRm48VD4sIGZuPzogU2NvcGVGbjxUPik6XG4gICAgICBUIHtcbiAgICBsZXQgbmFtZTogc3RyaW5nID0gbnVsbDtcbiAgICBpZiAoZm4gPT0gbnVsbCkge1xuICAgICAgLy8gQ2FsbGVkIHdpdGggb25seSAxIGFyZ3VtZW50LlxuICAgICAgaWYgKHR5cGVvZiBuYW1lT3JGbiAhPT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1BsZWFzZSBwcm92aWRlIGEgZnVuY3Rpb24gdG8gdGlkeSgpJyk7XG4gICAgICB9XG4gICAgICBmbiA9IG5hbWVPckZuO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBDYWxsZWQgd2l0aCAyIGFyZ3VtZW50cy5cbiAgICAgIGlmICh0eXBlb2YgbmFtZU9yRm4gIT09ICdzdHJpbmcnICYmICEobmFtZU9yRm4gaW5zdGFuY2VvZiBTdHJpbmcpKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdXaGVuIGNhbGxpbmcgd2l0aCB0d28gYXJndW1lbnRzLCB0aGUgZmlyc3QgYXJndW1lbnQgJyArXG4gICAgICAgICAgICAndG8gdGlkeSgpIG11c3QgYmUgYSBzdHJpbmcnKTtcbiAgICAgIH1cbiAgICAgIGlmICh0eXBlb2YgZm4gIT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ1doZW4gY2FsbGluZyB3aXRoIHR3byBhcmd1bWVudHMsIHRoZSAybmQgYXJndW1lbnQgJyArXG4gICAgICAgICAgICAndG8gdGlkeSgpIG11c3QgYmUgYSBmdW5jdGlvbicpO1xuICAgICAgfVxuICAgICAgbmFtZSA9IG5hbWVPckZuIGFzIHN0cmluZztcbiAgICAgIC8vIFRPRE8obnN0aG9yYXQsc21pbGtvdik6IERvIG9wZXJhdGlvbiBsb2dnaW5nIGFuZCBwZXJmb3JtYW5jZVxuICAgICAgLy8gcHJvZmlsaW5nLlxuICAgIH1cbiAgICBsZXQgcmVzdWx0OiBUO1xuICAgIHJldHVybiB0aGlzLnNjb3BlZFJ1bihcbiAgICAgICAgKCkgPT4gdGhpcy5zdGFydFNjb3BlKG5hbWUpLCAoKSA9PiB0aGlzLmVuZFNjb3BlKHJlc3VsdCksICgpID0+IHtcbiAgICAgICAgICByZXN1bHQgPSBmbigpO1xuICAgICAgICAgIGlmIChyZXN1bHQgaW5zdGFuY2VvZiBQcm9taXNlKSB7XG4gICAgICAgICAgICBjb25zb2xlLmVycm9yKCdDYW5ub3QgcmV0dXJuIGEgUHJvbWlzZSBpbnNpZGUgb2YgdGlkeS4nKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgICAgfSk7XG4gIH1cblxuICBwcml2YXRlIHNjb3BlZFJ1bjxUPihzdGFydDogKCkgPT4gdm9pZCwgZW5kOiAoKSA9PiB2b2lkLCBmOiAoKSA9PiBUKTogVCB7XG4gICAgc3RhcnQoKTtcbiAgICB0cnkge1xuICAgICAgY29uc3QgcmVzID0gZigpO1xuICAgICAgZW5kKCk7XG4gICAgICByZXR1cm4gcmVzO1xuICAgIH0gY2F0Y2ggKGV4KSB7XG4gICAgICBlbmQoKTtcbiAgICAgIHRocm93IGV4O1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgc3RhdGljIG5leHRUZW5zb3JJZCA9IDA7XG4gIHByaXZhdGUgbmV4dFRlbnNvcklkKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIEVuZ2luZS5uZXh0VGVuc29ySWQrKztcbiAgfVxuXG4gIHByaXZhdGUgc3RhdGljIG5leHRWYXJpYWJsZUlkID0gMDtcbiAgcHJpdmF0ZSBuZXh0VmFyaWFibGVJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBFbmdpbmUubmV4dFZhcmlhYmxlSWQrKztcbiAgfVxuXG4gIC8qKlxuICAgKiBUaGlzIG1ldGhvZCBpcyBjYWxsZWQgaW5zdGVhZCBvZiB0aGUgcHVibGljLWZhY2luZyB0ZW5zb3IuY2xvbmUoKSB3aGVuXG4gICAqIHNhdmluZyBhIHRlbnNvciBmb3IgYmFja3dhcmRzIHBhc3MuIEl0IG1ha2VzIHN1cmUgdG8gYWRkIHRoZSBjbG9uZVxuICAgKiBvcGVyYXRpb24gdG8gdGhlIHRhcGUgcmVnYXJkbGVzcyBvZiBiZWluZyBjYWxsZWQgaW5zaWRlIGEga2VybmVsXG4gICAqIGV4ZWN1dGlvbi5cbiAgICovXG4gIHByaXZhdGUgY2xvbmUoeDogVGVuc29yKTogVGVuc29yIHtcbiAgICBjb25zdCB5OiBUZW5zb3IgPSBFTkdJTkUucnVuS2VybmVsKElkZW50aXR5LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAge3h9IGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXApO1xuICAgIGNvbnN0IGlucHV0cyA9IHt4fTtcbiAgICBjb25zdCBncmFkID0gKGR5OiBUZW5zb3IpID0+ICh7XG4gICAgICB4OiAoKSA9PiB7XG4gICAgICAgIGNvbnN0IGR0eXBlID0gJ2Zsb2F0MzInO1xuICAgICAgICBjb25zdCBncmFkSW5wdXRzID0ge3g6IGR5fTtcbiAgICAgICAgY29uc3QgYXR0cnMgPSB7ZHR5cGV9O1xuXG4gICAgICAgIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgIENhc3QsIGdyYWRJbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFRlbnNvcjtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBjb25zdCBzYXZlZDogVGVuc29yW10gPSBbXTtcbiAgICB0aGlzLmFkZFRhcGVOb2RlKHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUubmFtZSwgaW5wdXRzLCBbeV0sIGdyYWQsIHNhdmVkLCB7fSk7XG4gICAgcmV0dXJuIHk7XG4gIH1cblxuICAvKipcbiAgICogRXhlY3V0ZSBhIGtlcm5lbCB3aXRoIHRoZSBnaXZlbiBuYW1lIGFuZCByZXR1cm4gdGhlIG91dHB1dCB0ZW5zb3IuXG4gICAqXG4gICAqIEBwYXJhbSBrZXJuZWxOYW1lIFRoZSBuYW1lIG9mIHRoZSBrZXJuZWwgdG8gZXhlY3V0ZS5cbiAgICogQHBhcmFtIGlucHV0cyBBIG1hcCBvZiBpbnB1dCBuYW1lcyB0byB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gYXR0cnMgQSBtYXAgb2YgYXR0cmlidXRlIG5hbWVzIHRvIHRoZWlyIHZhbHVlcy4gQW4gYXR0cmlidXRlIGlzIGFcbiAgICogICAgIHByaW1pdGl2ZSAobm9uLXRlbnNvcikgaW5wdXQgdG8gdGhlIGtlcm5lbC5cbiAgICogQHBhcmFtIGlucHV0c1RvU2F2ZSBBIGxpc3Qgb2YgdGVuc29ycywgaW5wdXRzIHRvIHNhdmUgZm9yIHRoZSBiYWNrcHJvcFxuICAgKiAgICAgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBvdXRwdXRzVG9TYXZlIEEgbGlzdCBvZiBib29sZWFucywgc3BlY2lmeWluZyB3aGljaCBvdXRwdXQgdG8gc2F2ZVxuICAgKiAgICAgZm9yIHRoZSBiYWNrcHJvcCBjb21wdXRhdGlvbi4gVGhlc2UgYXJlIGJvb2xlYW5zIHNpbmNlIHRoZSBvdXRwdXRcbiAgICogdGVuc29ycyBhcmUgbm90IHZpc2libGUgdG8gdGhlIHVzZXIuXG4gICAqL1xuICBydW5LZXJuZWw8VCBleHRlbmRzIFRlbnNvcnxUZW5zb3JbXT4oXG4gICAgICBrZXJuZWxOYW1lOiBzdHJpbmcsIGlucHV0czogTmFtZWRUZW5zb3JNYXAsIGF0dHJzPzogTmFtZWRBdHRyTWFwKTogVCB7XG4gICAgaWYgKHRoaXMuYmFja2VuZE5hbWUgPT0gbnVsbCkge1xuICAgICAgLy8gYmFja2VuZCBoYXMgbm90IGJlZW4gaW5pdGlhbGl6ZWQgeWV0IChiYWNrZW5kIGluaXRpYWxpemF0aW9uIGlzIGxhenlcbiAgICAgIC8vIGNhbiBiZSBkZWZlcnJlZCB1bnRpbCBhbiBvcC8ga2VybmVsIGlzIHJ1bikuXG4gICAgICAvLyBUaGUgYmVsb3cgZ2V0dGVyIGhhcyBzaWRlIGVmZmVjdHMgdGhhdCB3aWxsIHRyeSB0byBpbml0aWFsaXplIHRoZVxuICAgICAgLy8gYmFja2VuZCBhbmQgc2V0IHByb3BlcnRpZXMgbGlrZSB0aGlzLmJhY2tlbmROYW1lXG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgICB0aGlzLmJhY2tlbmQ7XG4gICAgfVxuICAgIGNvbnN0IGhhc0tlcm5lbCA9IGdldEtlcm5lbChrZXJuZWxOYW1lLCB0aGlzLmJhY2tlbmROYW1lKSAhPSBudWxsO1xuICAgIGlmICghaGFzS2VybmVsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEtlcm5lbCAnJHtrZXJuZWxOYW1lfScgbm90IHJlZ2lzdGVyZWQgZm9yIGJhY2tlbmQgJyR7XG4gICAgICAgICAgdGhpcy5iYWNrZW5kTmFtZX0nYCk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLnJ1bktlcm5lbEZ1bmMoe2tlcm5lbE5hbWUsIGlucHV0cywgYXR0cnN9KTtcbiAgfVxuXG4gIHByaXZhdGUgc2hvdWxkQ2hlY2tGb3JNZW1MZWFrcygpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5FTlYuZ2V0Qm9vbCgnSVNfVEVTVCcpO1xuICB9XG5cbiAgcHJpdmF0ZSBjaGVja0tlcm5lbEZvck1lbUxlYWsoXG4gICAgICBrZXJuZWxOYW1lOiBzdHJpbmcsIG51bURhdGFJZHNCZWZvcmU6IG51bWJlcixcbiAgICAgIG91dEluZm9zOiBUZW5zb3JJbmZvW10pOiB2b2lkIHtcbiAgICBjb25zdCBudW1EYXRhSWRzQWZ0ZXIgPSB0aGlzLmJhY2tlbmQubnVtRGF0YUlkcygpO1xuXG4gICAgLy8gQ291bnQgdGhlIG51bWJlciBvZiBkYXRhIGlkcyBhc3NvY2lhdGVkIHdpdGggdGhlIHJlc3VsdCBvZiB0aGUga2VybmVsLlxuICAgIGxldCBudW1PdXRwdXREYXRhSWRzID0gMDtcbiAgICBvdXRJbmZvcy5mb3JFYWNoKGluZm8gPT4ge1xuICAgICAgLy8gQ29tcGxleCBudW1iZXJzIGFsbG9jYXRlIDMgZGF0YSBpZHMsIG9uZSBmb3IgJ3JlYWwnLCBvbmUgZm9yXG4gICAgICAvLyAnaW1hZ2luYXJ5JywgYW5kIG9uZSBmb3IgdGhlIGNvbnRhaW5lciB0aGF0IGhvbGRzIHRoZSBmb3JtZXIgdHdvLlxuICAgICAgbnVtT3V0cHV0RGF0YUlkcyArPSAoaW5mby5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcgPyAzIDogMSk7XG4gICAgfSk7XG5cbiAgICAvLyBBY2NvdW50IGZvciB0aGUgbnVtYmVyIG9mIG1vdmVzIGR1cmluZyBrZXJuZWwgZXhlY3V0aW9uLiBBIFwiZGF0YSBtb3ZlXCJcbiAgICAvLyBjYW4gaGFwcGVuIGluIHRoZSBtaWRkbGUgb2YgYSBrZXJuZWwgZXhlY3V0aW9uLCBwbGFjaW5nIGEgbmV3IChrZXksdmFsdWUpXG4gICAgLy8gcGFpciBpbiB0aGUgZGF0YSBzdG9yYWdlLiBTaW5jZSBkYXRhIG1vdmVzIGhhdmUgbmV0IHplcm8gZWZmZWN0ICh3ZVxuICAgIC8vIGFsd2F5cyByZW1vdmUgdGhlIGRhdGEgZnJvbSB0aGUgb2xkIGJhY2tlbmQpLCB3ZSBoYXZlIHRvIGNhbmNlbCB0aGVtIG91dFxuICAgIC8vIHdoZW4gZGV0ZWN0aW5nIG1lbW9yeSBsZWFrcy5cbiAgICBjb25zdCBudW1Nb3ZlcyA9XG4gICAgICAgIHRoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2tbdGhpcy5zdGF0ZS5udW1EYXRhTW92ZXNTdGFjay5sZW5ndGggLSAxXTtcbiAgICBjb25zdCBkYXRhSWRzTGVha2VkID1cbiAgICAgICAgbnVtRGF0YUlkc0FmdGVyIC0gbnVtRGF0YUlkc0JlZm9yZSAtIG51bU91dHB1dERhdGFJZHMgLSBudW1Nb3ZlcztcbiAgICBpZiAoZGF0YUlkc0xlYWtlZCA+IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQmFja2VuZCAnJHt0aGlzLmJhY2tlbmROYW1lfScgaGFzIGFuIGludGVybmFsIG1lbW9yeSBsZWFrIGAgK1xuICAgICAgICAgIGAoJHtkYXRhSWRzTGVha2VkfSBkYXRhIGlkcykgYWZ0ZXIgcnVubmluZyAnJHtrZXJuZWxOYW1lfSdgKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogSW50ZXJuYWwgaGVscGVyIG1ldGhvZCB0byBleGVjdXRlIGEga2VybmVsIEZ1bmNcbiAgICpcbiAgICogVXNlIGBydW5LZXJuZWxgIHRvIGV4ZWN1dGUga2VybmVscyBmcm9tIG91dHNpZGUgb2YgZW5naW5lLlxuICAgKi9cbiAgcHJpdmF0ZSBydW5LZXJuZWxGdW5jPFQgZXh0ZW5kcyBUZW5zb3J8VGVuc29yW10sIEkgZXh0ZW5kcyBOYW1lZFRlbnNvck1hcD4oXG4gICAgICBrZXJuZWxQYXJhbXM6IFJlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPEk+fFxuICAgICAgQ3VzdG9tR3JhZEtlcm5lbEludm9jYXRpb248VCwgST4pOiBUIHtcbiAgICBsZXQgb3V0cHV0czogVGVuc29yW107XG4gICAgbGV0IHNhdmVkOiBUZW5zb3JbXSA9IFtdO1xuICAgIGNvbnN0IGlzVGFwZU9uID0gdGhpcy5pc1RhcGVPbigpO1xuXG4gICAgY29uc3Qgc3RhcnRpbmdCeXRlY291bnQgPSB0aGlzLnN0YXRlLm51bUJ5dGVzO1xuICAgIGNvbnN0IHN0YXJ0aW5nTnVtVGVuc29ycyA9IHRoaXMuc3RhdGUubnVtVGVuc29ycztcblxuICAgIGlmICh0aGlzLnNob3VsZENoZWNrRm9yTWVtTGVha3MoKSkge1xuICAgICAgdGhpcy5zdGF0ZS5udW1EYXRhTW92ZXNTdGFjay5wdXNoKDApO1xuICAgIH1cblxuICAgIGxldCBrZXJuZWxGdW5jOiAoKSA9PiBUZW5zb3JbXTtcbiAgICBpZiAodGhpcy5iYWNrZW5kTmFtZSA9PSBudWxsKSB7XG4gICAgICAvLyBiYWNrZW5kIGhhcyBub3QgYmVlbiBpbml0aWFsaXplZCB5ZXQgKGJhY2tlbmQgaW5pdGlhbGl6YXRpb24gaXMgbGF6eVxuICAgICAgLy8gY2FuIGJlIGRlZmVycmVkIHVudGlsIGFuIG9wLyBrZXJuZWwgaXMgcnVuKS5cbiAgICAgIC8vIFRoZSBiZWxvdyBnZXR0ZXIgaGFzIHNpZGUgZWZmZWN0cyB0aGF0IHdpbGwgdHJ5IHRvIGluaXRpYWxpemUgdGhlXG4gICAgICAvLyBiYWNrZW5kIGFuZCBzZXQgcHJvcGVydGllcyBsaWtlIHRoaXMuYmFja2VuZE5hbWVcbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW51c2VkLWV4cHJlc3Npb25cbiAgICAgIHRoaXMuYmFja2VuZDtcbiAgICB9XG5cbiAgICBsZXQgb3V0OiBUZW5zb3JJbmZvfFRlbnNvckluZm9bXTtcblxuICAgIGNvbnN0IGtlcm5lbE9yU2NvcGVOYW1lID0gaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbihrZXJuZWxQYXJhbXMpID9cbiAgICAgICAga2VybmVsUGFyYW1zLmtlcm5lbE5hbWUgOlxuICAgICAgICB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlICE9IG51bGwgPyB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLm5hbWUgOiAnJztcblxuICAgIC8vIENyZWF0ZSB0aGUga2VybmVsRnVuYyBmcm9tIGVpdGhlciBhIHJlZ2lzdGVyZWQga2VybmVsIE9SIHBhc3NlZCBpblxuICAgIC8vIGZvcndhcmQvYmFja3dhcmQgZnVuY3Rpb25zICh1c2VkIGJ5IGN1c3RvbSBncmFkKS4gSW4gdGhpcyBjb250ZXh0IGFcbiAgICAvLyBrZXJuZWxGdW5jIHdyYXBzIGEga2VybmVsIGltcGxlbWVudGF0aW9uIHdpdGggc29tZSBib29ra2VlcGluZy5cblxuICAgIGlmIChpc1JlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uKGtlcm5lbFBhcmFtcykpIHtcbiAgICAgIGNvbnN0IHtrZXJuZWxOYW1lLCBpbnB1dHMsIGF0dHJzfSA9IGtlcm5lbFBhcmFtcztcbiAgICAgIGlmICh0aGlzLmJhY2tlbmROYW1lID09IG51bGwpIHtcbiAgICAgICAgLy8gYmFja2VuZCBoYXMgbm90IGJlZW4gaW5pdGlhbGl6ZWQgeWV0IChiYWNrZW5kIGluaXRpYWxpemF0aW9uIGlzIGxhenlcbiAgICAgICAgLy8gY2FuIGJlIGRlZmVycmVkIHVudGlsIGFuIG9wLyBrZXJuZWwgaXMgcnVuKS5cbiAgICAgICAgLy8gVGhlIGJlbG93IGdldHRlciBoYXMgc2lkZSBlZmZlY3RzIHRoYXQgd2lsbCB0cnkgdG8gaW5pdGlhbGl6ZSB0aGVcbiAgICAgICAgLy8gYmFja2VuZCBhbmQgc2V0IHByb3BlcnRpZXMgbGlrZSB0aGlzLmJhY2tlbmROYW1lXG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW51c2VkLWV4cHJlc3Npb25cbiAgICAgICAgdGhpcy5iYWNrZW5kO1xuICAgICAgfVxuICAgICAgY29uc3Qga2VybmVsID0gZ2V0S2VybmVsKGtlcm5lbE5hbWUsIHRoaXMuYmFja2VuZE5hbWUpO1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAga2VybmVsICE9IG51bGwsXG4gICAgICAgICAgKCkgPT4gYENhbm5vdCBmaW5kIHJlZ2lzdGVyZWQga2VybmVsICcke2tlcm5lbE5hbWV9JyBmb3IgYmFja2VuZCAnJHtcbiAgICAgICAgICAgICAgdGhpcy5iYWNrZW5kTmFtZX0nYCk7XG5cbiAgICAgIGtlcm5lbEZ1bmMgPSAoKSA9PiB7XG4gICAgICAgIGNvbnN0IG51bURhdGFJZHNCZWZvcmUgPSB0aGlzLmJhY2tlbmQubnVtRGF0YUlkcygpO1xuICAgICAgICBvdXQgPSBrZXJuZWwua2VybmVsRnVuYyh7aW5wdXRzLCBhdHRycywgYmFja2VuZDogdGhpcy5iYWNrZW5kfSk7XG4gICAgICAgIGNvbnN0IG91dEluZm9zID0gQXJyYXkuaXNBcnJheShvdXQpID8gb3V0IDogW291dF07XG4gICAgICAgIGlmICh0aGlzLnNob3VsZENoZWNrRm9yTWVtTGVha3MoKSkge1xuICAgICAgICAgIHRoaXMuY2hlY2tLZXJuZWxGb3JNZW1MZWFrKGtlcm5lbE5hbWUsIG51bURhdGFJZHNCZWZvcmUsIG91dEluZm9zKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGNvbnN0IG91dFRlbnNvcnMgPSBvdXRJbmZvcy5tYXAoKG91dEluZm86IFRlbnNvckluZm98VGVuc29yKSA9PiB7XG4gICAgICAgICAgLy8gdG9kbyAoeWFzc29nYmEpIHJlbW92ZSB0aGlzIG9wdGlvbiAoVGVuc29yKSB3aGVuIG5vZGUgYmFja2VuZFxuICAgICAgICAgIC8vIG1ldGhvZHMgaGF2ZSBiZWVuIG1vZHVsYXJpemVkIGFuZCB0aGV5IGFsbCByZXR1cm4gdGVuc29ySW5mby5cbiAgICAgICAgICAvLyBUZW5zb3JJbmZvcyBkbyBub3QgaGF2ZSBhIHJhbmsgYXR0cmlidXRlLlxuICAgICAgICAgIGlmICgob3V0SW5mbyBhcyBUZW5zb3IpLnJhbmsgIT0gbnVsbCkge1xuICAgICAgICAgICAgcmV0dXJuIG91dEluZm8gYXMgVGVuc29yO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm4gdGhpcy5tYWtlVGVuc29yRnJvbVRlbnNvckluZm8ob3V0SW5mbyk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIFNhdmUgYW55IHJlcXVpcmVkIGlucHV0cyBhbmQgb3V0cHV0cy5cblxuICAgICAgICAvLyBEbyBub3Qgc2F2ZSB1bmxlc3Mgd2UgYXJlIHJlY29yZGluZyB0byB0aGUgdGFwZS4gT3RoZXJ3aXNlIGl0IHdvdWxkXG4gICAgICAgIC8vIGNhdXNlIGEgbWVtIGxlYWsgc2luY2UgdGhlcmUgd291bGQgYmUgbm8gYmFja3Byb3AgZm9yIHRoZXNlIHRlbnNvcnNcbiAgICAgICAgLy8gKHdoaWNoIHdvdWxkIG90aGVyd2lzZSBkaXNwb3NlIHRoZW0pLlxuICAgICAgICBpZiAoaXNUYXBlT24pIHtcbiAgICAgICAgICBjb25zdCB0ZW5zb3JzVG9TYXZlID1cbiAgICAgICAgICAgICAgdGhpcy5nZXRUZW5zb3JzRm9yR3JhZGllbnQoa2VybmVsTmFtZSwgaW5wdXRzLCBvdXRUZW5zb3JzKTtcbiAgICAgICAgICBzYXZlZCA9IHRoaXMuc2F2ZVRlbnNvcnNGb3JCYWNrd2FyZE1vZGUodGVuc29yc1RvU2F2ZSk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG91dFRlbnNvcnM7XG4gICAgICB9O1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB7Zm9yd2FyZEZ1bmN9ID0ga2VybmVsUGFyYW1zO1xuICAgICAgLy8gUnVubmluZyBhIGN1c3RvbUdyYWQgb3AuXG4gICAgICBjb25zdCBzYXZlRnVuYzogR3JhZFNhdmVGdW5jID0gKHRlbnNvcnMpID0+IHtcbiAgICAgICAgLy8gRG8gbm90IHNhdmUgdW5sZXNzIHdlIGFyZSByZWNvcmRpbmcgdG8gdGhlIHRhcGUuIE90aGVyd2lzZSBpdCB3b3VsZFxuICAgICAgICAvLyBjYXVzZSBhIG1lbSBsZWFrIHNpbmNlIHdlIHdvdWxkIG5ldmVyIHJ1biBiYWNrcHJvcCwgd2hpY2ggZGlzcG9zZXNcbiAgICAgICAgLy8gdGhlIGtlcHQgdGVuc29ycy5cbiAgICAgICAgaWYgKCFpc1RhcGVPbikge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBzYXZlZCA9IHRlbnNvcnMubWFwKHRlbnNvciA9PiB0aGlzLmtlZXAodGhpcy5jbG9uZSh0ZW5zb3IpKSk7XG4gICAgICB9O1xuXG4gICAgICBrZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBudW1EYXRhSWRzQmVmb3JlID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcbiAgICAgICAgb3V0ID0gdGhpcy50aWR5KCgpID0+IGZvcndhcmRGdW5jKHRoaXMuYmFja2VuZCwgc2F2ZUZ1bmMpKTtcbiAgICAgICAgY29uc3Qgb3V0cyA9IChBcnJheS5pc0FycmF5KG91dCkgPyBvdXQgOiBbb3V0XSkgYXMgVGVuc29yW107XG4gICAgICAgIGlmICh0aGlzLnNob3VsZENoZWNrRm9yTWVtTGVha3MoKSkge1xuICAgICAgICAgIC8vIFNjb3BlIG5hbWUgaXMgdXNlZCB0byBwcmludCBhIG1vcmUgaGVscGZ1bCBlcnJvciBtZXNzYWdlIGlmIG5lZWRlZC5cbiAgICAgICAgICB0aGlzLmNoZWNrS2VybmVsRm9yTWVtTGVhayhrZXJuZWxPclNjb3BlTmFtZSwgbnVtRGF0YUlkc0JlZm9yZSwgb3V0cyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG91dHM7XG4gICAgICB9O1xuICAgIH1cblxuICAgIC8vXG4gICAgLy8gUnVuIHRoZSBrZXJuZWxGdW5jLiBPcHRpb25hbGx5IHByb2ZpbGluZyBpdC5cbiAgICAvL1xuICAgIGNvbnN0IHtpbnB1dHMsIGF0dHJzfSA9IGtlcm5lbFBhcmFtcztcbiAgICBjb25zdCBiYWNrd2FyZHNGdW5jID0gaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbihrZXJuZWxQYXJhbXMpID9cbiAgICAgICAgbnVsbCA6XG4gICAgICAgIGtlcm5lbFBhcmFtcy5iYWNrd2FyZHNGdW5jO1xuXG4gICAgbGV0IGtlcm5lbFByb2ZpbGU6IEtlcm5lbFByb2ZpbGU7XG4gICAgdGhpcy5zY29wZWRSdW4oXG4gICAgICAgIC8vIFN0b3AgcmVjb3JkaW5nIHRvIGEgdGFwZSB3aGVuIHJ1bm5pbmcgYSBrZXJuZWwuXG4gICAgICAgICgpID0+IHRoaXMuc3RhdGUua2VybmVsRGVwdGgrKywgKCkgPT4gdGhpcy5zdGF0ZS5rZXJuZWxEZXB0aC0tLCAoKSA9PiB7XG4gICAgICAgICAgaWYgKCF0aGlzLkVOVi5nZXRCb29sKCdERUJVRycpICYmICF0aGlzLnN0YXRlLnByb2ZpbGluZykge1xuICAgICAgICAgICAgb3V0cHV0cyA9IGtlcm5lbEZ1bmMoKTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAga2VybmVsUHJvZmlsZSA9IHRoaXMucHJvZmlsZXIucHJvZmlsZUtlcm5lbChcbiAgICAgICAgICAgICAgICBrZXJuZWxPclNjb3BlTmFtZSwgaW5wdXRzLCAoKSA9PiBrZXJuZWxGdW5jKCkpO1xuICAgICAgICAgICAgaWYgKHRoaXMuRU5WLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgICAgICAgICAgdGhpcy5wcm9maWxlci5sb2dLZXJuZWxQcm9maWxlKGtlcm5lbFByb2ZpbGUpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgb3V0cHV0cyA9IGtlcm5lbFByb2ZpbGUub3V0cHV0cztcbiAgICAgICAgICB9XG4gICAgICAgIH0pO1xuXG4gICAgaWYgKGlzVGFwZU9uKSB7XG4gICAgICB0aGlzLmFkZFRhcGVOb2RlKFxuICAgICAgICAgIGtlcm5lbE9yU2NvcGVOYW1lLCBpbnB1dHMsIG91dHB1dHMsIGJhY2t3YXJkc0Z1bmMsIHNhdmVkLCBhdHRycyk7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMuc3RhdGUucHJvZmlsaW5nKSB7XG4gICAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUua2VybmVscy5wdXNoKHtcbiAgICAgICAgbmFtZToga2VybmVsT3JTY29wZU5hbWUsXG4gICAgICAgIGJ5dGVzQWRkZWQ6IHRoaXMuc3RhdGUubnVtQnl0ZXMgLSBzdGFydGluZ0J5dGVjb3VudCxcbiAgICAgICAgdG90YWxCeXRlc1NuYXBzaG90OiB0aGlzLnN0YXRlLm51bUJ5dGVzLFxuICAgICAgICB0ZW5zb3JzQWRkZWQ6IHRoaXMuc3RhdGUubnVtVGVuc29ycyAtIHN0YXJ0aW5nTnVtVGVuc29ycyxcbiAgICAgICAgdG90YWxUZW5zb3JzU25hcHNob3Q6IHRoaXMuc3RhdGUubnVtVGVuc29ycyxcbiAgICAgICAgaW5wdXRTaGFwZXM6IE9iamVjdC5rZXlzKGlucHV0cykubWFwKFxuICAgICAgICAgICAga2V5ID0+IGlucHV0c1trZXldICE9IG51bGwgPyBpbnB1dHNba2V5XS5zaGFwZSA6IG51bGwpLFxuICAgICAgICBvdXRwdXRTaGFwZXM6IG91dHB1dHMubWFwKGl0ZW0gPT4gaXRlbS5zaGFwZSksXG4gICAgICAgIGtlcm5lbFRpbWVNczoga2VybmVsUHJvZmlsZS50aW1lTXMsXG4gICAgICAgIGV4dHJhSW5mbzoga2VybmVsUHJvZmlsZS5leHRyYUluZm9cbiAgICAgIH0pO1xuICAgIH1cbiAgICByZXR1cm4gKEFycmF5LmlzQXJyYXkob3V0KSA/IG91dHB1dHMgOiBvdXRwdXRzWzBdKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIFNhdmVzIHRlbnNvcnMgdXNlZCBpbiBmb3J3YXJkIG1vZGUgZm9yIHVzZSBpbiBiYWNrd2FyZCBtb2RlLlxuICAgKlxuICAgKiBAcGFyYW0gdGVuc29ycyB0aGUgbGlzdCBvZiB0ZW5zb3JzIHRvIHNhdmUuXG4gICAqL1xuICBwcml2YXRlIHNhdmVUZW5zb3JzRm9yQmFja3dhcmRNb2RlKHRlbnNvcnM6IFRlbnNvcltdKTogVGVuc29yW10ge1xuICAgIGNvbnN0IHNhdmVkID0gdGVuc29ycy5tYXAodGVuc29yID0+IHRoaXMua2VlcCh0aGlzLmNsb25lKHRlbnNvcikpKTtcbiAgICByZXR1cm4gc2F2ZWQ7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBhIGxpc3Qgb2YgdGVuc29ycyB0byBzYXZlIGZvciBhIGdpdmVuIGdyYWRpZW50IGNhbGN1bGF0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0ga2VybmVsTmFtZSBuYW1lIG9mIGtlcm5lbCB0byBsb29rIHVwIGdyYWRpZW50IGZvci5cbiAgICogQHBhcmFtIGlucHV0cyBhIG1hcCBvZiBpbnB1dCB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gb3V0cHV0cyBhbiBhcnJheSBvZiBvdXRwdXQgdGVuc29ycyBmcm9tIGZvcndhcmQgbW9kZSBvZiBrZXJuZWwuXG4gICAqL1xuICBwcml2YXRlIGdldFRlbnNvcnNGb3JHcmFkaWVudChcbiAgICAgIGtlcm5lbE5hbWU6IHN0cmluZywgaW5wdXRzOiBOYW1lZFRlbnNvck1hcCxcbiAgICAgIG91dHB1dHM6IFRlbnNvcltdKTogVGVuc29yW118bnVsbCB7XG4gICAgY29uc3QgZ3JhZENvbmZpZyA9IGdldEdyYWRpZW50KGtlcm5lbE5hbWUpO1xuICAgIGlmIChncmFkQ29uZmlnICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IGlucHV0c1RvU2F2ZTogc3RyaW5nW10gPSBncmFkQ29uZmlnLmlucHV0c1RvU2F2ZSB8fCBbXTtcbiAgICAgIGNvbnN0IG91dHB1dHNUb1NhdmU6IGJvb2xlYW5bXSA9IGdyYWRDb25maWcub3V0cHV0c1RvU2F2ZSB8fCBbXTtcblxuICAgICAgLy8gSWYgc2F2ZUFsbElucHV0cyBpcyB0cnVlLCBhbGwgaW5wdXRzIHdpbGwgYmUgc2F2ZWQuIE90aGVyd2lzZSwgaW5wdXRzXG4gICAgICAvLyBzcGVjaWZpZWQgaW4gaW5wdXRzVG9TYXZlIHdpbGwgYmUgc2F2ZWQuXG4gICAgICBsZXQgaW5wdXRUZW5zb3JzVG9TYXZlOiBUZW5zb3JbXTtcbiAgICAgIGlmIChncmFkQ29uZmlnLnNhdmVBbGxJbnB1dHMpIHtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICBBcnJheS5pc0FycmF5KGlucHV0cyksXG4gICAgICAgICAgICAoKSA9PiAnc2F2ZUFsbElucHV0cyBpcyB0cnVlLCBleHBlY3RlZCBpbnB1dHMgdG8gYmUgYW4gYXJyYXkuJyk7XG5cbiAgICAgICAgaW5wdXRUZW5zb3JzVG9TYXZlID0gT2JqZWN0LmtleXMoaW5wdXRzKS5tYXAoKGtleSkgPT4gaW5wdXRzW2tleV0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgaW5wdXRUZW5zb3JzVG9TYXZlID0gaW5wdXRzVG9TYXZlLm1hcCgoaW5wdXROYW1lKSA9PiBpbnB1dHNbaW5wdXROYW1lXSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IG91dHB1dFRlbnNvcnNUb1NhdmU6IFRlbnNvcltdID1cbiAgICAgICAgICBvdXRwdXRzLmZpbHRlcigoXywgaSkgPT4gb3V0cHV0c1RvU2F2ZVtpXSk7XG5cbiAgICAgIHJldHVybiBpbnB1dFRlbnNvcnNUb1NhdmUuY29uY2F0KG91dHB1dFRlbnNvcnNUb1NhdmUpO1xuICAgIH1cbiAgICAvLyBXZSByZXR1cm4gYW4gZW1wdHkgbGlzdCByYXRoZXIgdGhhbiB0aHJvdyBhbiBlcnJvciBiZWNhdXNlIHRoZSBrZXJuZWwgd2VcbiAgICAvLyBhcmUgbG9va2luZyB1cCBtYXkgbm90IGFjdHVhbGx5IGJlIHJlbGV2YW50IHRvIGJhY2twcm9waW5nIHRocm91Z2ggdGhlXG4gICAgLy8gb3ZlcmFsbCBmdW5jdGlvblxuICAgIC8vXG4gICAgLy8gU2VlICdkb2VzIG5vdCBlcnJvciBpZiBpcnJlbGV2YW50IChwcnVuZWQpIG9wcyBhcmUgbWlzc2luZyBncmFkcycgdGVzdFxuICAgIC8vIGluIGdyYWRpZW50c190ZXN0LnRzIGZvciBhbiBleGFtcGxlLlxuICAgIHJldHVybiBbXTtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbnRlcm5hbCBtZXRob2QgdXNlZCBieSBwdWJsaWMgQVBJcyBmb3IgdGVuc29yIGNyZWF0aW9uLiBNYWtlcyBhIG5ld1xuICAgKiB0ZW5zb3Igd2l0aCB0aGUgcHJvdmlkZWQgc2hhcGUsIGR0eXBlIGFuZCB2YWx1ZXMuIEl0IGFsd2F5c1xuICAgKiBjcmVhdGVzIGEgbmV3IGRhdGEgaWQgYW5kIHdyaXRlcyB0aGUgdmFsdWVzIHRvIHRoZSB1bmRlcmx5aW5nIGJhY2tlbmQuXG4gICAqL1xuICBtYWtlVGVuc29yKFxuICAgICAgdmFsdWVzOiBEYXRhVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGJhY2tlbmQ/OiBLZXJuZWxCYWNrZW5kKTogVGVuc29yIHtcbiAgICBpZiAodmFsdWVzID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignVmFsdWVzIHBhc3NlZCB0byBlbmdpbmUubWFrZVRlbnNvcigpIGFyZSBudWxsJyk7XG4gICAgfVxuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGJhY2tlbmQgPSBiYWNrZW5kIHx8IHRoaXMuYmFja2VuZDtcbiAgICBsZXQgYmFja2VuZFZhbHMgPSB2YWx1ZXMgYXMgQmFja2VuZFZhbHVlcztcbiAgICBpZiAoZHR5cGUgPT09ICdzdHJpbmcnICYmIHV0aWwuaXNTdHJpbmcodmFsdWVzWzBdKSkge1xuICAgICAgYmFja2VuZFZhbHMgPSAodmFsdWVzIGFzIHN0cmluZ1tdKS5tYXAoZCA9PiB1dGlsLmVuY29kZVN0cmluZyhkKSk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGFJZCA9IGJhY2tlbmQud3JpdGUoYmFja2VuZFZhbHMsIHNoYXBlLCBkdHlwZSk7XG4gICAgY29uc3QgdCA9IG5ldyBUZW5zb3Ioc2hhcGUsIGR0eXBlLCBkYXRhSWQsIHRoaXMubmV4dFRlbnNvcklkKCkpO1xuICAgIHRoaXMudHJhY2tUZW5zb3IodCwgYmFja2VuZCk7XG5cbiAgICAvLyBDb3VudCBieXRlcyBmb3Igc3RyaW5nIHRlbnNvcnMuXG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKTtcbiAgICAgIGNvbnN0IG5ld0J5dGVzID0gYnl0ZXNGcm9tU3RyaW5nQXJyYXkoYmFja2VuZFZhbHMgYXMgVWludDhBcnJheVtdKTtcbiAgICAgIHRoaXMuc3RhdGUubnVtQnl0ZXMgKz0gbmV3Qnl0ZXMgLSBpbmZvLmJ5dGVzO1xuICAgICAgaW5mby5ieXRlcyA9IG5ld0J5dGVzO1xuICAgIH1cbiAgICByZXR1cm4gdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbnRlcm5hbCBtZXRob2QgdXNlZCBieSBiYWNrZW5kcy4gTWFrZXMgYSBuZXcgdGVuc29yXG4gICAqIHRoYXQgaXMgYSB3cmFwcGVyIGFyb3VuZCBhbiBleGlzdGluZyBkYXRhIGlkLiBJdCBkb2Vzbid0IGNyZWF0ZVxuICAgKiBhIG5ldyBkYXRhIGlkLCBvbmx5IGluY3JlbWVudHMgdGhlIHJlZiBjb3VudCB1c2VkIGluIG1lbW9yeSB0cmFja2luZy5cbiAgICogQGRlcHJlY2F0ZWRcbiAgICovXG4gIG1ha2VUZW5zb3JGcm9tRGF0YUlkKFxuICAgIGRhdGFJZDogRGF0YUlkLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICBiYWNrZW5kPzogS2VybmVsQmFja2VuZCk6IFRlbnNvciB7XG4gICAgZHR5cGUgPSBkdHlwZSB8fCAnZmxvYXQzMic7XG4gICAgY29uc3QgdGVuc29ySW5mbzogVGVuc29ySW5mbyA9IHtkYXRhSWQsIHNoYXBlLCBkdHlwZX07XG4gICAgcmV0dXJuIHRoaXMubWFrZVRlbnNvckZyb21UZW5zb3JJbmZvKHRlbnNvckluZm8sIGJhY2tlbmQpO1xuICB9XG5cbiAgLyoqXG4gICAqIEludGVybmFsIG1ldGhvZCB1c2VkIGJ5IGJhY2tlbmRzLiBNYWtlcyBhIG5ldyB0ZW5zb3IgdGhhdCBpcyBhIHdyYXBwZXJcbiAgICogYXJvdW5kIGFuIGV4aXN0aW5nIGRhdGEgaWQgaW4gVGVuc29ySW5mby4gSXQgZG9lc24ndCBjcmVhdGUgYSBuZXcgZGF0YSBpZCxcbiAgICogb25seSBpbmNyZW1lbnRzIHRoZSByZWYgY291bnQgdXNlZCBpbiBtZW1vcnkgdHJhY2tpbmcuXG4gICAqL1xuICBtYWtlVGVuc29yRnJvbVRlbnNvckluZm8odGVuc29ySW5mbzogVGVuc29ySW5mbywgYmFja2VuZD86IEtlcm5lbEJhY2tlbmQpOlxuICAgICAgVGVuc29yIHtcbiAgICBjb25zdCB7ZGF0YUlkLCBzaGFwZSwgZHR5cGV9ID0gdGVuc29ySW5mbztcbiAgICBjb25zdCB0ID0gbmV3IFRlbnNvcihzaGFwZSwgZHR5cGUsIGRhdGFJZCwgdGhpcy5uZXh0VGVuc29ySWQoKSk7XG4gICAgdGhpcy50cmFja1RlbnNvcih0LCBiYWNrZW5kKTtcbiAgICByZXR1cm4gdDtcbiAgfVxuXG4gIG1ha2VWYXJpYWJsZShcbiAgICAgIGluaXRpYWxWYWx1ZTogVGVuc29yLCB0cmFpbmFibGUgPSB0cnVlLCBuYW1lPzogc3RyaW5nLFxuICAgICAgZHR5cGU/OiBEYXRhVHlwZSk6IFZhcmlhYmxlIHtcbiAgICBuYW1lID0gbmFtZSB8fCB0aGlzLm5leHRWYXJpYWJsZUlkKCkudG9TdHJpbmcoKTtcbiAgICBpZiAoZHR5cGUgIT0gbnVsbCAmJiBkdHlwZSAhPT0gaW5pdGlhbFZhbHVlLmR0eXBlKSB7XG4gICAgICBpbml0aWFsVmFsdWUgPSBpbml0aWFsVmFsdWUuY2FzdChkdHlwZSk7XG4gICAgfVxuICAgIGNvbnN0IHYgPSBuZXcgVmFyaWFibGUoaW5pdGlhbFZhbHVlLCB0cmFpbmFibGUsIG5hbWUsIHRoaXMubmV4dFRlbnNvcklkKCkpO1xuICAgIGlmICh0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdi5uYW1lXSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFZhcmlhYmxlIHdpdGggbmFtZSAke3YubmFtZX0gd2FzIGFscmVhZHkgcmVnaXN0ZXJlZGApO1xuICAgIH1cbiAgICB0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdi5uYW1lXSA9IHY7XG4gICAgdGhpcy5pbmNSZWYodiwgdGhpcy5iYWNrZW5kKTtcbiAgICByZXR1cm4gdjtcbiAgfVxuXG4gIHRyYWNrVGVuc29yKGE6IFRlbnNvciwgYmFja2VuZDogS2VybmVsQmFja2VuZCk6IHZvaWQge1xuICAgIHRoaXMuc3RhdGUubnVtVGVuc29ycysrO1xuICAgIGlmIChhLmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgdGhpcy5zdGF0ZS5udW1TdHJpbmdUZW5zb3JzKys7XG4gICAgfVxuICAgIC8vIEJ5dGVzIGZvciBjb21wbGV4IG51bWJlcnMgYXJlIGNvdW50ZWQgYnkgdGhlaXIgY29tcG9uZW50cy4gQnl0ZXMgZm9yXG4gICAgLy8gc3RyaW5nIHRlbnNvcnMgYXJlIGNvdW50ZWQgd2hlbiB3cml0aW5nIHZhbHVlcy5cbiAgICBsZXQgYnl0ZXMgPSAwO1xuICAgIGlmIChhLmR0eXBlICE9PSAnY29tcGxleDY0JyAmJiBhLmR0eXBlICE9PSAnc3RyaW5nJykge1xuICAgICAgYnl0ZXMgPSBhLnNpemUgKiB1dGlsLmJ5dGVzUGVyRWxlbWVudChhLmR0eXBlKTtcbiAgICB9XG4gICAgdGhpcy5zdGF0ZS5udW1CeXRlcyArPSBieXRlcztcblxuICAgIGlmICghdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmhhcyhhLmRhdGFJZCkpIHtcbiAgICAgIHRoaXMuc3RhdGUubnVtRGF0YUJ1ZmZlcnMrKztcbiAgICAgIHRoaXMuc3RhdGUudGVuc29ySW5mby5zZXQoYS5kYXRhSWQsIHtcbiAgICAgICAgYmFja2VuZDogYmFja2VuZCB8fCB0aGlzLmJhY2tlbmQsXG4gICAgICAgIGR0eXBlOiBhLmR0eXBlLFxuICAgICAgICBzaGFwZTogYS5zaGFwZSxcbiAgICAgICAgYnl0ZXNcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmICghKGEgaW5zdGFuY2VvZiBWYXJpYWJsZSkpIHtcbiAgICAgIHRoaXMudHJhY2soYSk7XG4gICAgfVxuICB9XG5cbiAgLy8gVHJhY2sgdGhlIHRlbnNvciBieSBkYXRhSWQgYW5kIGluY3JlYXNlIHRoZSByZWZDb3VudCBmb3IgdGhlIGRhdGFJZCBpbiB0aGVcbiAgLy8gYmFja2VuZC5cbiAgLy8gVE9ETyhweXUxMDA1NSk6IFRoaXMgaXMgY3VycmVudGx5IHVzZWQgYnkgbWFrZVZhcmlhYmxlIG1ldGhvZCwgdG8gaW5jcmVhc2VcbiAgLy8gcmVmQ291bnQgb24gdGhlIGJhY2tlbmQgZm9yIHRoZSBkYXRhSWQuIEl0IGNhbiBwb3RlbnRpYWxseSBiZSByZXBsYWNlZCB3aXRoXG4gIC8vIElkZW50aXR5IG9wIGluZGVhZCBvZiBjYWxsaW5nIGJhY2tlbmQgZGlyZWN0bHkuXG4gIGluY1JlZihhOiBUZW5zb3IsIGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQpOiB2b2lkIHtcbiAgICB0aGlzLnRyYWNrVGVuc29yKGEsIGJhY2tlbmQpO1xuICAgIHRoaXMuYmFja2VuZC5pbmNSZWYoYS5kYXRhSWQpO1xuICB9XG5cbiAgcmVtb3ZlRGF0YUlkKGRhdGFJZDogRGF0YUlkLCBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kKSB7XG4gICAgaWYgKHRoaXMuc3RhdGUudGVuc29ySW5mby5oYXMoZGF0YUlkKSAmJlxuICAgICAgICB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCkuYmFja2VuZCA9PT0gYmFja2VuZCkge1xuICAgICAgdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmRlbGV0ZShkYXRhSWQpO1xuICAgICAgdGhpcy5zdGF0ZS5udW1EYXRhQnVmZmVycy0tO1xuICAgIH1cbiAgfVxuICBkaXNwb3NlVGVuc29yKGE6IFRlbnNvcik6IHZvaWQge1xuICAgIGlmICghdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmhhcyhhLmRhdGFJZCkpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoYS5kYXRhSWQpO1xuXG4gICAgdGhpcy5zdGF0ZS5udW1UZW5zb3JzLS07XG4gICAgaWYgKGEuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICB0aGlzLnN0YXRlLm51bVN0cmluZ1RlbnNvcnMtLTtcbiAgICAgIHRoaXMuc3RhdGUubnVtQnl0ZXMgLT0gaW5mby5ieXRlcztcbiAgICB9XG4gICAgLy8gRG9uJ3QgY291bnQgYnl0ZXMgZm9yIGNvbXBsZXggbnVtYmVycyBhcyB0aGV5IGFyZSBjb3VudGVkIGJ5IHRoZWlyXG4gICAgLy8gY29tcG9uZW50cy5cbiAgICBpZiAoYS5kdHlwZSAhPT0gJ2NvbXBsZXg2NCcgJiYgYS5kdHlwZSAhPT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IGJ5dGVzID0gYS5zaXplICogdXRpbC5ieXRlc1BlckVsZW1lbnQoYS5kdHlwZSk7XG4gICAgICB0aGlzLnN0YXRlLm51bUJ5dGVzIC09IGJ5dGVzO1xuICAgIH1cblxuICAgIC8vIFJlbW92ZSB0aGUgcmVmZXJlbmNlIHRvIGRhdGFJZCBpZiBiYWNrZW5kIGRpc3Bvc2UgdGhlIGRhdGEgc3VjY2Vzc2Z1bGx5XG4gICAgaWYgKGluZm8uYmFja2VuZC5kaXNwb3NlRGF0YShhLmRhdGFJZCkpIHtcbiAgICAgIHRoaXMucmVtb3ZlRGF0YUlkKGEuZGF0YUlkLCBpbmZvLmJhY2tlbmQpO1xuICAgIH1cblxuICAgIC8vIFRPRE8obnN0aG9yYXQpOiBDb25zdHJ1Y3QgYW4gZXJyb3IgYW5kIHNhdmUgdGhlIHN0YWNrIHRyYWNlIGZvclxuICAgIC8vIGRlYnVnZ2luZyB3aGVuIGluIGRlYnVnIG1vZGUuIENyZWF0aW5nIGEgc3RhY2sgdHJhY2UgaXMgdG9vIGV4cGVuc2l2ZVxuICAgIC8vIHRvIGRvIHVuY29uZGl0aW9uYWxseS5cbiAgfVxuXG4gIGRpc3Bvc2VWYXJpYWJsZXMoKTogdm9pZCB7XG4gICAgZm9yIChjb25zdCB2YXJOYW1lIGluIHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlcykge1xuICAgICAgY29uc3QgdiA9IHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlc1t2YXJOYW1lXTtcbiAgICAgIHRoaXMuZGlzcG9zZVZhcmlhYmxlKHYpO1xuICAgIH1cbiAgfVxuXG4gIGRpc3Bvc2VWYXJpYWJsZSh2OiBWYXJpYWJsZSk6IHZvaWQge1xuICAgIHRoaXMuZGlzcG9zZVRlbnNvcih2KTtcbiAgICBpZiAodGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzW3YubmFtZV0gIT0gbnVsbCkge1xuICAgICAgZGVsZXRlIHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlc1t2Lm5hbWVdO1xuICAgIH1cbiAgfVxuXG4gIG1lbW9yeSgpOiBNZW1vcnlJbmZvIHtcbiAgICBjb25zdCBpbmZvID0gdGhpcy5iYWNrZW5kLm1lbW9yeSgpIGFzIE1lbW9yeUluZm87XG4gICAgaW5mby5udW1UZW5zb3JzID0gdGhpcy5zdGF0ZS5udW1UZW5zb3JzO1xuICAgIGluZm8ubnVtRGF0YUJ1ZmZlcnMgPSB0aGlzLnN0YXRlLm51bURhdGFCdWZmZXJzO1xuICAgIGluZm8ubnVtQnl0ZXMgPSB0aGlzLnN0YXRlLm51bUJ5dGVzO1xuICAgIGlmICh0aGlzLnN0YXRlLm51bVN0cmluZ1RlbnNvcnMgPiAwKSB7XG4gICAgICBpbmZvLnVucmVsaWFibGUgPSB0cnVlO1xuICAgICAgaWYgKGluZm8ucmVhc29ucyA9PSBudWxsKSB7XG4gICAgICAgIGluZm8ucmVhc29ucyA9IFtdO1xuICAgICAgfVxuICAgICAgaW5mby5yZWFzb25zLnB1c2goXG4gICAgICAgICAgJ01lbW9yeSB1c2FnZSBieSBzdHJpbmcgdGVuc29ycyBpcyBhcHByb3hpbWF0ZSAnICtcbiAgICAgICAgICAnKDIgYnl0ZXMgcGVyIGNoYXJhY3RlciknKTtcbiAgICB9XG4gICAgcmV0dXJuIGluZm87XG4gIH1cblxuICBhc3luYyBwcm9maWxlKHF1ZXJ5OiAoKSA9PiAoVGVuc29yQ29udGFpbmVyIHwgUHJvbWlzZTxUZW5zb3JDb250YWluZXI+KSk6XG4gICAgICBQcm9taXNlPFByb2ZpbGVJbmZvPiB7XG4gICAgdGhpcy5zdGF0ZS5wcm9maWxpbmcgPSB0cnVlO1xuXG4gICAgY29uc3Qgc3RhcnRCeXRlcyA9IHRoaXMuc3RhdGUubnVtQnl0ZXM7XG4gICAgY29uc3Qgc3RhcnROdW1UZW5zb3JzID0gdGhpcy5zdGF0ZS5udW1UZW5zb3JzO1xuXG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLmtlcm5lbHMgPSBbXTtcbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUucmVzdWx0ID0gYXdhaXQgcXVlcnkoKTtcblxuICAgIHRoaXMuc3RhdGUucHJvZmlsaW5nID0gZmFsc2U7XG5cbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUucGVha0J5dGVzID0gTWF0aC5tYXgoXG4gICAgICAgIC4uLnRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5rZXJuZWxzLm1hcChkID0+IGQudG90YWxCeXRlc1NuYXBzaG90KSk7XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLm5ld0J5dGVzID0gdGhpcy5zdGF0ZS5udW1CeXRlcyAtIHN0YXJ0Qnl0ZXM7XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLm5ld1RlbnNvcnMgPVxuICAgICAgICB0aGlzLnN0YXRlLm51bVRlbnNvcnMgLSBzdGFydE51bVRlbnNvcnM7XG4gICAgZm9yIChjb25zdCBrZXJuZWwgb2YgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLmtlcm5lbHMpIHtcbiAgICAgIGtlcm5lbC5rZXJuZWxUaW1lTXMgPSBhd2FpdCBrZXJuZWwua2VybmVsVGltZU1zO1xuICAgICAga2VybmVsLmV4dHJhSW5mbyA9IGF3YWl0IGtlcm5lbC5leHRyYUluZm87XG4gICAgfVxuICAgIHJldHVybiB0aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGU7XG4gIH1cblxuICBpc1RhcGVPbigpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5zdGF0ZS5ncmFkaWVudERlcHRoID4gMCAmJiB0aGlzLnN0YXRlLmtlcm5lbERlcHRoID09PSAwO1xuICB9XG5cbiAgcHJpdmF0ZSBhZGRUYXBlTm9kZShcbiAgICAgIGtlcm5lbE5hbWU6IHN0cmluZywgaW5wdXRzOiBOYW1lZFRlbnNvck1hcCwgb3V0cHV0czogVGVuc29yW10sXG4gICAgICBncmFkaWVudHNGdW5jOiBHcmFkRnVuYywgc2F2ZWQ6IFRlbnNvcltdLCBhdHRyczogTmFtZWRBdHRyTWFwKTogdm9pZCB7XG4gICAgY29uc3QgdGFwZU5vZGU6IFRhcGVOb2RlID1cbiAgICAgICAge2lkOiB0aGlzLnN0YXRlLm5leHRUYXBlTm9kZUlkKyssIGtlcm5lbE5hbWUsIGlucHV0cywgb3V0cHV0cywgc2F2ZWR9O1xuXG4gICAgY29uc3QgZ3JhZENvbmZpZyA9IGdldEdyYWRpZW50KGtlcm5lbE5hbWUpO1xuICAgIGlmIChncmFkQ29uZmlnICE9IG51bGwpIHtcbiAgICAgIGdyYWRpZW50c0Z1bmMgPSBncmFkQ29uZmlnLmdyYWRGdW5jO1xuICAgIH1cbiAgICBpZiAoZ3JhZGllbnRzRnVuYyAhPSBudWxsKSB7XG4gICAgICB0YXBlTm9kZS5ncmFkaWVudCA9IChkeXM6IFRlbnNvcltdKSA9PiB7XG4gICAgICAgIC8vIFRPRE8oc21pbGtvdik6IFRvIG9wdGltaXplIGJhY2stcHJvcCwgcGFzcyBkeXMgdGhhdCBhcmUgbm90IHVzZWQgaW5cbiAgICAgICAgLy8gdGhlIGJhY2twcm9wIGdyYXBoIHRvIHRoZSB1c2VyIGFzIG51bGwgaW5zdGVhZCBvZiB6ZXJvc1xuICAgICAgICBkeXMgPSBkeXMubWFwKChkeSwgaSkgPT4ge1xuICAgICAgICAgIGlmIChkeSA9PSBudWxsKSB7XG4gICAgICAgICAgICBjb25zdCBvdXRwdXQgPSBvdXRwdXRzW2ldO1xuICAgICAgICAgICAgY29uc3QgdmFscyA9IHV0aWwubWFrZVplcm9zVHlwZWRBcnJheShvdXRwdXQuc2l6ZSwgb3V0cHV0LmR0eXBlKTtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLm1ha2VUZW5zb3IodmFscywgb3V0cHV0LnNoYXBlLCBvdXRwdXQuZHR5cGUpO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm4gZHk7XG4gICAgICAgIH0pO1xuICAgICAgICAvLyBHcmFkIGZ1bmN0aW9ucyBvZiBvcHMgd2l0aCBzaW5nbGUgb3V0cHV0cyBleHBlY3QgYSBkeSwgd2hpbGUgb3BzXG4gICAgICAgIC8vIHdpdGggbXVsdGlwbGUgb3V0cHV0cyBleHBlY3QgZHlzIChhcnJheSBvZiBkeSkuXG4gICAgICAgIHJldHVybiBncmFkaWVudHNGdW5jKGR5cy5sZW5ndGggPiAxID8gZHlzIDogZHlzWzBdLCBzYXZlZCwgYXR0cnMpO1xuICAgICAgfTtcbiAgICB9XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVUYXBlLnB1c2godGFwZU5vZGUpO1xuICB9XG5cbiAga2VlcDxUIGV4dGVuZHMgVGVuc29yPihyZXN1bHQ6IFQpOiBUIHtcbiAgICByZXN1bHQua2VwdCA9IHRydWU7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgc3RhcnRUYXBlKCkge1xuICAgIGlmICh0aGlzLnN0YXRlLmdyYWRpZW50RGVwdGggPT09IDApIHtcbiAgICAgIHRoaXMuc3RhdGUuYWN0aXZlVGFwZSA9IFtdO1xuICAgIH1cbiAgICB0aGlzLnN0YXRlLmdyYWRpZW50RGVwdGgrKztcbiAgfVxuXG4gIHByaXZhdGUgZW5kVGFwZSgpIHtcbiAgICB0aGlzLnN0YXRlLmdyYWRpZW50RGVwdGgtLTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdGFydCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIGVuZFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgc3RhcnRTY29wZShuYW1lPzogc3RyaW5nKSB7XG4gICAgY29uc3Qgc2NvcGVJbmZvOiBTY29wZVN0YXRlID0ge1xuICAgICAgdHJhY2s6IFtdLFxuICAgICAgbmFtZTogJ3VubmFtZWQgc2NvcGUnLFxuICAgICAgaWQ6IHRoaXMuc3RhdGUubmV4dFNjb3BlSWQrK1xuICAgIH07XG4gICAgaWYgKG5hbWUpIHtcbiAgICAgIHNjb3BlSW5mby5uYW1lID0gbmFtZTtcbiAgICB9XG4gICAgdGhpcy5zdGF0ZS5zY29wZVN0YWNrLnB1c2goc2NvcGVJbmZvKTtcbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlID0gc2NvcGVJbmZvO1xuICB9XG5cbiAgLyoqXG4gICAqIEVuZCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIHN0YXJ0U2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBlbmRTY29wZShyZXN1bHQ/OiBUZW5zb3JDb250YWluZXIpIHtcbiAgICBjb25zdCB0ZW5zb3JzVG9UcmFja0luUGFyZW50ID0gZ2V0VGVuc29yc0luQ29udGFpbmVyKHJlc3VsdCk7XG4gICAgY29uc3QgdGVuc29yc1RvVHJhY2tJblBhcmVudFNldCA9XG4gICAgICAgIG5ldyBTZXQodGVuc29yc1RvVHJhY2tJblBhcmVudC5tYXAodCA9PiB0LmlkKSk7XG5cbiAgICAvLyBEaXNwb3NlIHRoZSBhcnJheXMgdHJhY2tlZCBpbiB0aGlzIHNjb3BlLlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS50cmFjay5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgdGVuc29yID0gdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS50cmFja1tpXTtcbiAgICAgIGlmICghdGVuc29yLmtlcHQgJiYgIXRlbnNvcnNUb1RyYWNrSW5QYXJlbnRTZXQuaGFzKHRlbnNvci5pZCkpIHtcbiAgICAgICAgdGVuc29yLmRpc3Bvc2UoKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCBvbGRTY29wZSA9IHRoaXMuc3RhdGUuc2NvcGVTdGFjay5wb3AoKTtcbiAgICB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlID0gdGhpcy5zdGF0ZS5zY29wZVN0YWNrLmxlbmd0aCA9PT0gMCA/XG4gICAgICAgIG51bGwgOlxuICAgICAgICB0aGlzLnN0YXRlLnNjb3BlU3RhY2tbdGhpcy5zdGF0ZS5zY29wZVN0YWNrLmxlbmd0aCAtIDFdO1xuXG4gICAgLy8gVHJhY2sgdGhlIGN1cnJlbnQgcmVzdWx0IGluIHRoZSBwYXJlbnQgc2NvcGUuXG4gICAgdGVuc29yc1RvVHJhY2tJblBhcmVudC5mb3JFYWNoKHRlbnNvciA9PiB7XG4gICAgICAvLyBPbmx5IHRyYWNrIHRoZSB0ZW5zb3IgaWYgd2FzIGFsbG9jYXRlZCBpbiB0aGUgaW5uZXIgc2NvcGUgYW5kIGlzIG5vdFxuICAgICAgLy8gZ2xvYmFsbHkga2VwdC5cbiAgICAgIGlmICghdGVuc29yLmtlcHQgJiYgdGVuc29yLnNjb3BlSWQgPT09IG9sZFNjb3BlLmlkKSB7XG4gICAgICAgIHRoaXMudHJhY2sodGVuc29yKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIGdyYWRpZW50cyBvZiBgZmAgd2l0aCByZXNwZWN0IHRvIGVhY2ggb2YgdGhlIGB4c2AuIFRoZSBncmFkaWVudHNcbiAgICogcmV0dXJuZWQgYXJlIG9mIHRoZSBzYW1lIGxlbmd0aCBhcyBgeHNgLCBidXQgc29tZSBtaWdodCBiZSBudWxsIGlmIGBmYFxuICAgKiB3YXMgbm90IGEgZnVuY3Rpb24gb2YgdGhhdCBgeGAuIEl0IGFsc28gdGFrZXMgb3B0aW9uYWwgZHkgdG8gbXVsdGlwbHkgdGhlXG4gICAqIGdyYWRpZW50LCB3aGljaCBkZWZhdWx0cyB0byBgMWAuXG4gICAqL1xuICBncmFkaWVudHM8VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgICBmOiAoKSA9PiBULCB4czogVGVuc29yW10sIGR5PzogVCxcbiAgICAgIGFsbG93Tm9HcmFkaWVudHMgPSBmYWxzZSk6IHt2YWx1ZTogVCwgZ3JhZHM6IFRlbnNvcltdfSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHhzLmxlbmd0aCA+IDAsICgpID0+ICdncmFkaWVudHMoKSByZWNlaXZlZCBhbiBlbXB0eSBsaXN0IG9mIHhzLicpO1xuICAgIGlmIChkeSAhPSBudWxsICYmIGR5LmR0eXBlICE9PSAnZmxvYXQzMicpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgZHkgbXVzdCBoYXZlICdmbG9hdDMyJyBkdHlwZSwgYnV0IGhhcyAnJHtkeS5kdHlwZX0nYCk7XG4gICAgfVxuXG4gICAgY29uc3QgeSA9IHRoaXMuc2NvcGVkUnVuKFxuICAgICAgICAoKSA9PiB0aGlzLnN0YXJ0VGFwZSgpLCAoKSA9PiB0aGlzLmVuZFRhcGUoKSxcbiAgICAgICAgKCkgPT4gdGhpcy50aWR5KCdmb3J3YXJkJywgZikpO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHkgaW5zdGFuY2VvZiBUZW5zb3IsXG4gICAgICAgICgpID0+ICdUaGUgcmVzdWx0IHkgcmV0dXJuZWQgYnkgZigpIG11c3QgYmUgYSB0ZW5zb3IuJyk7XG4gICAgLy8gRmlsdGVyIG91dCB0aGUgbm9kZXMgdGhhdCBkb24ndCBjb25uZWN0IHggPT4geS5cbiAgICBjb25zdCBmaWx0ZXJlZFRhcGUgPSBnZXRGaWx0ZXJlZE5vZGVzWFRvWSh0aGlzLnN0YXRlLmFjdGl2ZVRhcGUsIHhzLCB5KTtcbiAgICBpZiAoIWFsbG93Tm9HcmFkaWVudHMgJiYgZmlsdGVyZWRUYXBlLmxlbmd0aCA9PT0gMCAmJiB4cy5sZW5ndGggPiAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ0Nhbm5vdCBjb21wdXRlIGdyYWRpZW50IG9mIHk9Zih4KSB3aXRoIHJlc3BlY3QgdG8geC4gTWFrZSBzdXJlICcgK1xuICAgICAgICAgICd0aGF0IHRoZSBmIHlvdSBwYXNzZWQgZW5jbG9zZXMgYWxsIG9wZXJhdGlvbnMgdGhhdCBsZWFkIGZyb20geCAnICtcbiAgICAgICAgICAndG8geS4nKTtcbiAgICB9XG5cbiAgICByZXR1cm4gdGhpcy50aWR5KCdiYWNrd2FyZCcsICgpID0+IHtcbiAgICAgIGNvbnN0IGFjY3VtdWxhdGVkR3JhZGllbnRNYXA6IHtbdGVuc29ySWQ6IG51bWJlcl06IFRlbnNvcn0gPSB7fTtcbiAgICAgIGFjY3VtdWxhdGVkR3JhZGllbnRNYXBbeS5pZF0gPSAoZHkgPT0gbnVsbCkgPyBvbmVzKHkuc2hhcGUpIDogZHk7XG5cbiAgICAgIC8vIEJhY2twcm9wIGdyYWRpZW50cyB0aHJvdWdoIHRoZSBmaWx0ZXJlZCBub2Rlcy5cbiAgICAgIGJhY2twcm9wYWdhdGVHcmFkaWVudHMoXG4gICAgICAgICAgYWNjdW11bGF0ZWRHcmFkaWVudE1hcCwgZmlsdGVyZWRUYXBlLFxuICAgICAgICAgIC8vIFBhc3MgdGhlIHRpZHkgZnVuY3Rpb24gdG8gYXZvaWQgY2lyY3VsYXIgZGVwIHdpdGggYHRhcGUudHNgLlxuICAgICAgICAgIGYgPT4gdGhpcy50aWR5KGYgYXMgU2NvcGVGbjxUZW5zb3I+KSxcbiAgICAgICAgICAvLyBQYXNzIGFuIGFkZCBmdW5jdGlvbiB0byBhdm9pZGUgYSBjaXJjdWxhciBkZXAgd2l0aCBgdGFwZS50c2AuXG4gICAgICAgICAgYWRkKTtcbiAgICAgIGNvbnN0IGdyYWRzID0geHMubWFwKHggPT4gYWNjdW11bGF0ZWRHcmFkaWVudE1hcFt4LmlkXSk7XG5cbiAgICAgIGlmICh0aGlzLnN0YXRlLmdyYWRpZW50RGVwdGggPT09IDApIHtcbiAgICAgICAgLy8gVGhpcyBtZWFucyB0aGF0IHdlIGFyZSBub3QgY29tcHV0aW5nIGhpZ2hlci1vcmRlciBncmFkaWVudHNcbiAgICAgICAgLy8gYW5kIGNhbiBjbGVhbiB1cCB0aGUgdGFwZS5cbiAgICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVUYXBlLmZvckVhY2gobm9kZSA9PiB7XG4gICAgICAgICAgZm9yIChjb25zdCB0ZW5zb3Igb2Ygbm9kZS5zYXZlZCkge1xuICAgICAgICAgICAgdGVuc29yLmRpc3Bvc2UoKTtcbiAgICAgICAgICB9XG4gICAgICAgIH0pO1xuICAgICAgICB0aGlzLnN0YXRlLmFjdGl2ZVRhcGUgPSBudWxsO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHt2YWx1ZTogeSwgZ3JhZHN9O1xuICAgIH0pO1xuICB9XG5cbiAgY3VzdG9tR3JhZDxUIGV4dGVuZHMgVGVuc29yPihmOiBDdXN0b21HcmFkaWVudEZ1bmM8VD4pOlxuICAgICAgKC4uLmFyZ3M6IEFycmF5PFRlbnNvcnxHcmFkU2F2ZUZ1bmM+KSA9PiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdXRpbC5pc0Z1bmN0aW9uKGYpLFxuICAgICAgICAoKSA9PiAnVGhlIGYgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikgbXVzdCBiZSBhIGZ1bmN0aW9uLicpO1xuICAgIHJldHVybiAoLi4uaW5wdXRzOiBUZW5zb3JbXSk6IFQgPT4ge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgaW5wdXRzLmV2ZXJ5KHQgPT4gdCBpbnN0YW5jZW9mIFRlbnNvciksXG4gICAgICAgICAgKCkgPT4gJ1RoZSBhcmdzIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpKHgxLCB4MiwuLi4pIG11c3QgYWxsIGJlICcgK1xuICAgICAgICAgICAgICAndGVuc29ycycpO1xuXG4gICAgICBsZXQgcmVzOiB7XG4gICAgICAgIHZhbHVlOiBULFxuICAgICAgICBncmFkRnVuYzogKGR5OiBULCBzYXZlZDogVGVuc29yW10pID0+IFRlbnNvciB8IFRlbnNvcltdLFxuICAgICAgfTtcbiAgICAgIGNvbnN0IGlucHV0TWFwOiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICAgICAgaW5wdXRzLmZvckVhY2goKGlucHV0LCBpKSA9PiB7XG4gICAgICAgIGlucHV0TWFwW2ldID0gaW5wdXQ7XG4gICAgICB9KTtcblxuICAgICAgY29uc3QgZm9yd2FyZEZ1bmM6IEZvcndhcmRGdW5jPFQ+ID0gKF8sIHNhdmUpID0+IHtcbiAgICAgICAgcmVzID0gZiguLi5bLi4uaW5wdXRzLCBzYXZlXSk7XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgcmVzLnZhbHVlIGluc3RhbmNlb2YgVGVuc29yLFxuICAgICAgICAgICAgKCkgPT4gJ1RoZSBmdW5jdGlvbiBmIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpIG11c3QgcmV0dXJuIGFuICcgK1xuICAgICAgICAgICAgICAgICdvYmplY3Qgd2hlcmUgYG9iai52YWx1ZWAgaXMgYSB0ZW5zb3InKTtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICB1dGlsLmlzRnVuY3Rpb24ocmVzLmdyYWRGdW5jKSxcbiAgICAgICAgICAgICgpID0+ICdUaGUgZnVuY3Rpb24gZiBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSBtdXN0IHJldHVybiBhbiAnICtcbiAgICAgICAgICAgICAgICAnb2JqZWN0IHdoZXJlIGBvYmouZ3JhZEZ1bmNgIGlzIGEgZnVuY3Rpb24uJyk7XG4gICAgICAgIHJldHVybiByZXMudmFsdWU7XG4gICAgICB9O1xuXG4gICAgICBjb25zdCBiYWNrd2FyZHNGdW5jID0gKGR5OiBULCBzYXZlZDogVGVuc29yW10pID0+IHtcbiAgICAgICAgY29uc3QgZ3JhZFJlcyA9IHJlcy5ncmFkRnVuYyhkeSwgc2F2ZWQpO1xuICAgICAgICBjb25zdCBncmFkczogVGVuc29yW10gPSBBcnJheS5pc0FycmF5KGdyYWRSZXMpID8gZ3JhZFJlcyA6IFtncmFkUmVzXTtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICBncmFkcy5sZW5ndGggPT09IGlucHV0cy5sZW5ndGgsXG4gICAgICAgICAgICAoKSA9PiAnVGhlIGZ1bmN0aW9uIGYgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikgbXVzdCByZXR1cm4gYW4gJyArXG4gICAgICAgICAgICAgICAgJ29iamVjdCB3aGVyZSBgb2JqLmdyYWRGdW5jYCBpcyBhIGZ1bmN0aW9uIHRoYXQgcmV0dXJucyAnICtcbiAgICAgICAgICAgICAgICAndGhlIHNhbWUgbnVtYmVyIG9mIHRlbnNvcnMgYXMgaW5wdXRzIHBhc3NlZCB0byBmKC4uLikuJyk7XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgZ3JhZHMuZXZlcnkodCA9PiB0IGluc3RhbmNlb2YgVGVuc29yKSxcbiAgICAgICAgICAgICgpID0+ICdUaGUgZnVuY3Rpb24gZiBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSBtdXN0IHJldHVybiBhbiAnICtcbiAgICAgICAgICAgICAgICAnb2JqZWN0IHdoZXJlIGBvYmouZ3JhZEZ1bmNgIGlzIGEgZnVuY3Rpb24gdGhhdCByZXR1cm5zICcgK1xuICAgICAgICAgICAgICAgICdhIGxpc3Qgb2Ygb25seSB0ZW5zb3JzLicpO1xuICAgICAgICBjb25zdCBncmFkTWFwOiB7W2tleTogc3RyaW5nXTogKCkgPT4gVGVuc29yfSA9IHt9O1xuICAgICAgICBncmFkcy5mb3JFYWNoKChncmFkLCBpKSA9PiB7XG4gICAgICAgICAgZ3JhZE1hcFtpXSA9ICgpID0+IGdyYWQ7XG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gZ3JhZE1hcDtcbiAgICAgIH07XG5cbiAgICAgIHJldHVybiB0aGlzLnJ1bktlcm5lbEZ1bmMoe1xuICAgICAgICBmb3J3YXJkRnVuYyxcbiAgICAgICAgYmFja3dhcmRzRnVuYyxcbiAgICAgICAgaW5wdXRzOiBpbnB1dE1hcCxcbiAgICAgIH0pO1xuICAgIH07XG4gIH1cblxuICByZWFkU3luYyhkYXRhSWQ6IERhdGFJZCk6IEJhY2tlbmRWYWx1ZXMge1xuICAgIC8vIFJvdXRlIHRoZSByZWFkIHRvIHRoZSBjb3JyZWN0IGJhY2tlbmQuXG4gICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKTtcbiAgICByZXR1cm4gaW5mby5iYWNrZW5kLnJlYWRTeW5jKGRhdGFJZCk7XG4gIH1cbiAgcmVhZChkYXRhSWQ6IERhdGFJZCk6IFByb21pc2U8QmFja2VuZFZhbHVlcz4ge1xuICAgIC8vIFJvdXRlIHRoZSByZWFkIHRvIHRoZSBjb3JyZWN0IGJhY2tlbmQuXG4gICAgY29uc3QgaW5mbyA9IHRoaXMuc3RhdGUudGVuc29ySW5mby5nZXQoZGF0YUlkKTtcbiAgICByZXR1cm4gaW5mby5iYWNrZW5kLnJlYWQoZGF0YUlkKTtcbiAgfVxuXG4gIHJlYWRUb0dQVShkYXRhSWQ6IERhdGFJZCwgb3B0aW9ucz86IERhdGFUb0dQVU9wdGlvbnMpOiBHUFVEYXRhIHtcbiAgICAvLyBSb3V0ZSB0aGUgcmVhZCB0byB0aGUgY29ycmVjdCBiYWNrZW5kLlxuICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCk7XG4gICAgcmV0dXJuIGluZm8uYmFja2VuZC5yZWFkVG9HUFUoZGF0YUlkLCBvcHRpb25zKTtcbiAgfVxuXG4gIGFzeW5jIHRpbWUocXVlcnk6ICgpID0+IHZvaWQpOiBQcm9taXNlPFRpbWluZ0luZm8+IHtcbiAgICBjb25zdCBzdGFydCA9IG5vdygpO1xuICAgIGNvbnN0IHRpbWluZ0luZm8gPSBhd2FpdCB0aGlzLmJhY2tlbmQudGltZShxdWVyeSkgYXMgVGltaW5nSW5mbztcbiAgICB0aW1pbmdJbmZvLndhbGxNcyA9IG5vdygpIC0gc3RhcnQ7XG4gICAgcmV0dXJuIHRpbWluZ0luZm87XG4gIH1cblxuICAvKipcbiAgICogVHJhY2tzIGEgVGVuc29yIGluIHRoZSBjdXJyZW50IHNjb3BlIHRvIGJlIGF1dG9tYXRpY2FsbHkgY2xlYW5lZCB1cFxuICAgKiB3aGVuIHRoZSBjdXJyZW50IHNjb3BlIGVuZHMsIGFuZCByZXR1cm5zIHRoZSB2YWx1ZS5cbiAgICpcbiAgICogQHBhcmFtIHJlc3VsdCBUaGUgVGVuc29yIHRvIHRyYWNrIGluIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgKi9cbiAgcHJpdmF0ZSB0cmFjazxUIGV4dGVuZHMgVGVuc29yPihyZXN1bHQ6IFQpOiBUIHtcbiAgICBpZiAodGhpcy5zdGF0ZS5hY3RpdmVTY29wZSAhPSBudWxsKSB7XG4gICAgICByZXN1bHQuc2NvcGVJZCA9IHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUuaWQ7XG4gICAgICB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLnRyYWNrLnB1c2gocmVzdWx0KTtcbiAgICB9XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgZ2V0IHJlZ2lzdGVyZWRWYXJpYWJsZXMoKTogTmFtZWRWYXJpYWJsZU1hcCB7XG4gICAgcmV0dXJuIHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlcztcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXNldHMgdGhlIGVuZ2luZSBzdGF0ZS4gUmVtb3ZlcyBhbGwgYmFja2VuZHMgYnV0IGRvZXMgbm90IHJlbW92ZVxuICAgKiByZWdpc3RlcmVkIGJhY2tlbmQgZmFjdG9yaWVzLlxuICAgKi9cbiAgcmVzZXQoKTogdm9pZCB7XG4gICAgLy8gTWFrZSBhbnkgcGVuZGluZyBwcm9taXNlIG9ic29sZXRlLlxuICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0SWQrKztcblxuICAgIHRoaXMuc3RhdGUuZGlzcG9zZSgpO1xuICAgIHRoaXMuRU5WLnJlc2V0KCk7XG4gICAgdGhpcy5zdGF0ZSA9IG5ldyBFbmdpbmVTdGF0ZSgpO1xuXG4gICAgZm9yIChjb25zdCBiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5KSB7XG4gICAgICB0aGlzLmRpc3Bvc2VSZWdpc3RlcmVkS2VybmVscyhiYWNrZW5kTmFtZSk7XG4gICAgICB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXS5kaXNwb3NlKCk7XG4gICAgICBkZWxldGUgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV07XG4gICAgfVxuICAgIHRoaXMuYmFja2VuZE5hbWUgPSBudWxsO1xuICAgIHRoaXMuYmFja2VuZEluc3RhbmNlID0gbnVsbDtcbiAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdCA9IG51bGw7XG4gIH1cbn1cblxuZnVuY3Rpb24gb25lcyhzaGFwZTogbnVtYmVyW10pOiBUZW5zb3Ige1xuICBjb25zdCB2YWx1ZXMgPSBtYWtlT25lc1R5cGVkQXJyYXkoc2l6ZUZyb21TaGFwZShzaGFwZSksICdmbG9hdDMyJyk7XG4gIHJldHVybiBFTkdJTkUubWFrZVRlbnNvcih2YWx1ZXMsIHNoYXBlLCAnZmxvYXQzMicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0T3JNYWtlRW5naW5lKCk6IEVuZ2luZSB7XG4gIGNvbnN0IG5zID0gZ2V0R2xvYmFsTmFtZXNwYWNlKCkgYXMgdW5rbm93biBhcyB7X3RmZW5naW5lOiBFbmdpbmV9O1xuICBpZiAobnMuX3RmZW5naW5lID09IG51bGwpIHtcbiAgICBjb25zdCBlbnZpcm9ubWVudCA9IG5ldyBFbnZpcm9ubWVudChucyk7XG4gICAgbnMuX3RmZW5naW5lID0gbmV3IEVuZ2luZShlbnZpcm9ubWVudCk7XG4gIH1cbiAgc2V0RW52aXJvbm1lbnRHbG9iYWwobnMuX3RmZW5naW5lLkVOVik7XG5cbiAgLy8gVGVsbCB0aGUgY3VycmVudCB0ZW5zb3IgaW50ZXJmYWNlIHRoYXQgdGhlIGdsb2JhbCBlbmdpbmUgaXMgcmVzcG9uc2libGVcbiAgLy8gZm9yIHRyYWNraW5nLlxuICBzZXRUZW5zb3JUcmFja2VyKCgpID0+IG5zLl90ZmVuZ2luZSk7XG4gIHJldHVybiBucy5fdGZlbmdpbmU7XG59XG5cbmV4cG9ydCBjb25zdCBFTkdJTkUgPSBnZXRPck1ha2VFbmdpbmUoKTtcblxuLyoqXG4gKiBBIGltcGxlbWVudGF0aW9uIG9mIHRoZSBhZGQgb3AgZm9yIHVzZSB3aXRoaW4gZW5naW5lIGFuZCB0YXBlLlxuICpcbiAqIFRoaXMgYWxsb3dzIHVzIHRvIGF2b2lkIGEgY2lyY3VsYXIgZGVwZW5kZW5jeSBiZXR3ZWVuIGFkZC50cyBhbmQgZW5naW5lLlxuICogSXQgaXMgZXhwb3J0ZWQgdG8gYmUgYXZhaWxhYmxlIGluIHRhcGUgdGVzdHMuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhZGQoYTogVGVuc29yLCBiOiBUZW5zb3IpOiBUZW5zb3Ige1xuICAvLyBXZSBkdXBsaWNhdGUgQWRkIGhlcmUgdG8gYXZvaWQgYSBjaXJjdWxhciBkZXBlbmRlbmN5IHdpdGggYWRkLnRzLlxuICBjb25zdCBpbnB1dHMgPSB7YSwgYn07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKEFkZCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXApO1xufVxuIl19