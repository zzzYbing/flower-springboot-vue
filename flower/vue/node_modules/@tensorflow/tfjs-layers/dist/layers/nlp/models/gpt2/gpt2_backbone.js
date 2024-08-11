/**
 * @license
 * Copyright 2023 Google LLC.
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
/**
 *  Base class for Backbone models.
 */
/* Original source: keras_nlp/models/gpt2/gpt2_backbone.py */
import { serialization } from '@tensorflow/tfjs-core';
import { RandomNormal } from '../../../../initializers';
import { input } from '../../../../exports';
import { Embedding } from '../../../embeddings';
import { PositionEmbedding } from '../../modeling/position_embedding';
import { add } from '../../../../exports_layers';
import { Dropout } from '../../../core';
import { TransformerDecoder } from '../../modeling/transformer_decoder';
import { getActivation } from '../../../../activations';
import { LayerNormalization } from '../../../normalization';
import { Backbone } from '../backbone';
function gpt2KernelInitializer(stddev = 0.02) {
    return new RandomNormal({ stddev });
}
/**
 * GPT-2 core network with hyperparameters.
 *
 * This network implements a Transformer-based decoder network,
 * Generative Pretrained Transformer-2 (GPT-2), as described in
 * ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
 * It includes the embedding lookups and transformer layers.
 *
 * The default constructor gives a fully customizable, randomly initialized
 * GPT-2 model with any number of layers, heads, and embedding
 * dimensions. To load preset architectures and weights, use the `fromPreset`
 * constructor.
 *
 * Disclaimer: Pre-trained models are provided on an "as is" basis, without
 * warranties or conditions of any kind. The underlying model is provided by a
 * third party and subject to a separate license, available
 * [here](https://github.com/openai/gpt-2).
 *
 *
 * Example usage:
 * ```js
 * const tokenIds = tf.ones([1, 12]), dtype="int32");
 * const paddingMask = tf.tensor(
 *  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], 'int32');
 *
 * # Pretrained GPT-2 decoder.
 * model = GPT2Backbone.fromPreset("gpt2_base_en");
 * model.apply(inputData, {paddingMask});
 *
 * # Randomly initialized GPT-2 decoder with custom config.
 * model = kerasNlp.models.GPT2Backbone({
 *     vocabularySize: 50257,
 *     numLayers: 12,
 *     numHeads: 12,
 *     hiddenDim: 768,
 *     intermediateDim: 3072,
 *     maxSequenceLength: 1024,
 * });
 * model.apply(inputData, {paddingMask});
 * ```
 */
class GPT2Backbone extends Backbone {
    constructor(args) {
        var _a, _b, _c, _d;
        args.dropout = (_a = args.dropout) !== null && _a !== void 0 ? _a : 0.1;
        args.maxSequenceLength = (_b = args.maxSequenceLength) !== null && _b !== void 0 ? _b : 1024;
        // Inputs
        const tokenIds = input({ shape: [null], dtype: 'int32', name: 'token_ids' });
        const paddingMask = input({ shape: [null], dtype: 'int32', name: 'padding_mask' });
        // Embed tokens, positions.
        const tokenEmbedding = new Embedding({
            inputDim: args.vocabularySize,
            outputDim: args.hiddenDim,
            embeddingsInitializer: gpt2KernelInitializer(0.01),
            name: 'token_embedding',
        }).apply(tokenIds);
        const positionEmbedding = new PositionEmbedding({
            initializer: gpt2KernelInitializer(0.02),
            sequenceLength: args.maxSequenceLength,
            name: 'position_embedding',
        }).apply(tokenEmbedding);
        // Sum and apply dropout to embeddings.
        let x = add({ name: 'embeddings_add' })
            .apply([tokenEmbedding, positionEmbedding]);
        x = new Dropout({ rate: args.dropout, name: 'embeddings_dropout' })
            .apply(x);
        // Apply successive transformer decoder blocks.
        for (let i = 0; i < args.numLayers; i++) {
            x = new TransformerDecoder({
                intermediateDim: args.intermediateDim,
                numHeads: args.numHeads,
                dropout: args.dropout,
                layerNormEpsilon: 1e-05,
                // TODO(pforderique): Implement gelu.
                activation: getActivation('relu'),
                kernelInitializer: gpt2KernelInitializer(0.02),
                normalizeFirst: true,
                name: `transformer_layer_${i}`,
            }).apply(x, { decoderPaddingMask: paddingMask });
        }
        const sequenceOutput = new LayerNormalization({
            name: 'layer_norm',
            axis: -1,
            epsilon: 1e-05,
            dtype: 'float32',
        }).apply(x);
        // Instantiate using Functional API Model constructor.
        super({
            inputs: [tokenIds, paddingMask],
            outputs: sequenceOutput,
            name: 'gpt2_backbone'
        });
        this.vocabularySize = args.vocabularySize;
        this.numLayers = args.numLayers;
        this.numHeads = args.numHeads;
        this.hiddenDim = args.hiddenDim;
        this.intermediateDim = args.intermediateDim;
        this.dropout = (_c = args.dropout) !== null && _c !== void 0 ? _c : 0.1;
        this.maxSequenceLength = (_d = args.maxSequenceLength) !== null && _d !== void 0 ? _d : 1024;
    }
    getConfig() {
        const config = {
            vocabularySize: this.vocabularySize,
            numLayers: this.numLayers,
            numHeads: this.numHeads,
            hiddenDim: this.hiddenDim,
            intermediateDim: this.intermediateDim,
            dropout: this.dropout,
            maxSequenceLength: this.maxSequenceLength,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    get tokenEmbedding() {
        return this.getLayer('token_embedding');
    }
}
/** @nocollapse */
GPT2Backbone.className = 'GPT2Backbone';
export { GPT2Backbone };
serialization.registerClass(GPT2Backbone);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3B0Ml9iYWNrYm9uZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvbmxwL21vZGVscy9ncHQyL2dwdDJfYmFja2JvbmUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUg7O0dBRUc7QUFFSCw2REFBNkQ7QUFDN0QsT0FBTyxFQUFFLGFBQWEsRUFBRSxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sRUFBRSxZQUFZLEVBQUUsTUFBTSwwQkFBMEIsQ0FBQztBQUN4RCxPQUFPLEVBQUUsS0FBSyxFQUFFLE1BQU0scUJBQXFCLENBQUM7QUFDNUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxNQUFNLHFCQUFxQixDQUFDO0FBRWhELE9BQU8sRUFBRSxpQkFBaUIsRUFBRSxNQUFNLG1DQUFtQyxDQUFDO0FBQ3RFLE9BQU8sRUFBRSxHQUFHLEVBQUUsTUFBTSw0QkFBNEIsQ0FBQztBQUNqRCxPQUFPLEVBQUUsT0FBTyxFQUFFLE1BQU0sZUFBZSxDQUFDO0FBQ3hDLE9BQU8sRUFBRSxrQkFBa0IsRUFBRSxNQUFNLG9DQUFvQyxDQUFDO0FBQ3hFLE9BQU8sRUFBRSxhQUFhLEVBQUUsTUFBTSx5QkFBeUIsQ0FBQztBQUN4RCxPQUFPLEVBQUUsa0JBQWtCLEVBQUUsTUFBTSx3QkFBd0IsQ0FBQztBQUM1RCxPQUFPLEVBQUUsUUFBUSxFQUFFLE1BQU0sYUFBYSxDQUFDO0FBRXZDLFNBQVMscUJBQXFCLENBQUMsTUFBTSxHQUFHLElBQUk7SUFDMUMsT0FBTyxJQUFJLFlBQVksQ0FBQyxFQUFDLE1BQU0sRUFBQyxDQUFDLENBQUM7QUFDcEMsQ0FBQztBQTZDRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXdDRztBQUNILE1BQWEsWUFBYSxTQUFRLFFBQVE7SUFZeEMsWUFBWSxJQUFzQjs7UUFDaEMsSUFB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QUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogIEJhc2UgY2xhc3MgZm9yIEJhY2tib25lIG1vZGVscy5cbiAqL1xuXG4vKiBPcmlnaW5hbCBzb3VyY2U6IGtlcmFzX25scC9tb2RlbHMvZ3B0Mi9ncHQyX2JhY2tib25lLnB5ICovXG5pbXBvcnQgeyBzZXJpYWxpemF0aW9uIH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHsgUmFuZG9tTm9ybWFsIH0gZnJvbSAnLi4vLi4vLi4vLi4vaW5pdGlhbGl6ZXJzJztcbmltcG9ydCB7IGlucHV0IH0gZnJvbSAnLi4vLi4vLi4vLi4vZXhwb3J0cyc7XG5pbXBvcnQgeyBFbWJlZGRpbmcgfSBmcm9tICcuLi8uLi8uLi9lbWJlZGRpbmdzJztcbmltcG9ydCB7IFN5bWJvbGljVGVuc29yIH0gZnJvbSAnLi4vLi4vLi4vLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7IFBvc2l0aW9uRW1iZWRkaW5nIH0gZnJvbSAnLi4vLi4vbW9kZWxpbmcvcG9zaXRpb25fZW1iZWRkaW5nJztcbmltcG9ydCB7IGFkZCB9IGZyb20gJy4uLy4uLy4uLy4uL2V4cG9ydHNfbGF5ZXJzJztcbmltcG9ydCB7IERyb3BvdXQgfSBmcm9tICcuLi8uLi8uLi9jb3JlJztcbmltcG9ydCB7IFRyYW5zZm9ybWVyRGVjb2RlciB9IGZyb20gJy4uLy4uL21vZGVsaW5nL3RyYW5zZm9ybWVyX2RlY29kZXInO1xuaW1wb3J0IHsgZ2V0QWN0aXZhdGlvbiB9IGZyb20gJy4uLy4uLy4uLy4uL2FjdGl2YXRpb25zJztcbmltcG9ydCB7IExheWVyTm9ybWFsaXphdGlvbiB9IGZyb20gJy4uLy4uLy4uL25vcm1hbGl6YXRpb24nO1xuaW1wb3J0IHsgQmFja2JvbmUgfSBmcm9tICcuLi9iYWNrYm9uZSc7XG5cbmZ1bmN0aW9uIGdwdDJLZXJuZWxJbml0aWFsaXplcihzdGRkZXYgPSAwLjAyKSB7XG4gIHJldHVybiBuZXcgUmFuZG9tTm9ybWFsKHtzdGRkZXZ9KTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBHUFQyQmFja2JvbmVBcmdzICB7XG4gIC8qKlxuICAgKiBJbnRlZ2VyLiBUaGUgc2l6ZSBvZiB0aGUgdG9rZW4gdm9jYWJ1bGFyeS5cbiAgICovXG4gIHZvY2FidWxhcnlTaXplOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEludGVnZXIuIFRoZSBudW1iZXIgb2YgdHJhbnNmb3JtZXIgbGF5ZXJzLlxuICAgKi9cbiAgbnVtTGF5ZXJzOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEludGVnZXIuIFRoZSBudW1iZXIgb2YgYXR0ZW50aW9uIGhlYWRzIGZvciBlYWNoIHRyYW5zZm9ybWVyLlxuICAgKiBUaGUgaGlkZGVuIHNpemUgbXVzdCBiZSBkaXZpc2libGUgYnkgdGhlIG51bWJlciBvZiBhdHRlbnRpb24gaGVhZHMuXG4gICAqL1xuICBudW1IZWFkczogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBJbnRlZ2VyLiBUaGUgc2l6ZSBvZiB0aGUgdHJhbnNmb3JtZXIgZW5jb2RpbmcgYW5kIHBvb2xlciBsYXllcnMuXG4gICAqL1xuICBoaWRkZW5EaW06IG51bWJlcjtcblxuICAvKipcbiAgICogSW50ZWdlci4gVGhlIG91dHB1dCBkaW1lbnNpb24gb2YgdGhlIGZpcnN0IERlbnNlIGxheWVyIGluIGEgdHdvLWxheWVyXG4gICAqIGZlZWRmb3J3YXJkIG5ldHdvcmsgZm9yIGVhY2ggdHJhbnNmb3JtZXIuXG4gICAqL1xuICBpbnRlcm1lZGlhdGVEaW06IG51bWJlcjtcblxuICAvKipcbiAgICogRmxvYXQuIERyb3BvdXQgcHJvYmFiaWxpdHkgZm9yIHRoZSBUcmFuc2Zvcm1lciBlbmNvZGVyLlxuICAgKiBEZWZhdWx0cyB0byAwLjIuXG4gICAqL1xuICBkcm9wb3V0PzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBJbnRlZ2VyLiBUaGUgbWF4aW11bSBzZXF1ZW5jZSBsZW5ndGggdGhhdCB0aGlzIGVuY29kZXIgY2FuIGNvbnN1bWUuXG4gICAqIElmIGBudWxsYCwgYG1heFNlcXVlbmNlTGVuZ3RoYCB1c2VzIHRoZSB2YWx1ZSBmcm9tIHNlcXVlbmNlIGxlbmd0aC5cbiAgICogVGhpcyBkZXRlcm1pbmVzIHRoZSB2YXJpYWJsZSBzaGFwZSBmb3IgcG9zaXRpb25hbCBlbWJlZGRpbmdzLlxuICAgKiBEZWZhdWx0cyB0byAxMDI0LlxuICAgKi9cbiAgbWF4U2VxdWVuY2VMZW5ndGg/OiBudW1iZXI7XG59XG5cbi8qKlxuICogR1BULTIgY29yZSBuZXR3b3JrIHdpdGggaHlwZXJwYXJhbWV0ZXJzLlxuICpcbiAqIFRoaXMgbmV0d29yayBpbXBsZW1lbnRzIGEgVHJhbnNmb3JtZXItYmFzZWQgZGVjb2RlciBuZXR3b3JrLFxuICogR2VuZXJhdGl2ZSBQcmV0cmFpbmVkIFRyYW5zZm9ybWVyLTIgKEdQVC0yKSwgYXMgZGVzY3JpYmVkIGluXG4gKiBbXCJMYW5ndWFnZSBNb2RlbHMgYXJlIFVuc3VwZXJ2aXNlZCBNdWx0aXRhc2sgTGVhcm5lcnNcIl0oaHR0cHM6Ly9jZG4ub3BlbmFpLmNvbS9iZXR0ZXItbGFuZ3VhZ2UtbW9kZWxzL2xhbmd1YWdlX21vZGVsc19hcmVfdW5zdXBlcnZpc2VkX211bHRpdGFza19sZWFybmVycy5wZGYpLlxuICogSXQgaW5jbHVkZXMgdGhlIGVtYmVkZGluZyBsb29rdXBzIGFuZCB0cmFuc2Zvcm1lciBsYXllcnMuXG4gKlxuICogVGhlIGRlZmF1bHQgY29uc3RydWN0b3IgZ2l2ZXMgYSBmdWxseSBjdXN0b21pemFibGUsIHJhbmRvbWx5IGluaXRpYWxpemVkXG4gKiBHUFQtMiBtb2RlbCB3aXRoIGFueSBudW1iZXIgb2YgbGF5ZXJzLCBoZWFkcywgYW5kIGVtYmVkZGluZ1xuICogZGltZW5zaW9ucy4gVG8gbG9hZCBwcmVzZXQgYXJjaGl0ZWN0dXJlcyBhbmQgd2VpZ2h0cywgdXNlIHRoZSBgZnJvbVByZXNldGBcbiAqIGNvbnN0cnVjdG9yLlxuICpcbiAqIERpc2NsYWltZXI6IFByZS10cmFpbmVkIG1vZGVscyBhcmUgcHJvdmlkZWQgb24gYW4gXCJhcyBpc1wiIGJhc2lzLCB3aXRob3V0XG4gKiB3YXJyYW50aWVzIG9yIGNvbmRpdGlvbnMgb2YgYW55IGtpbmQuIFRoZSB1bmRlcmx5aW5nIG1vZGVsIGlzIHByb3ZpZGVkIGJ5IGFcbiAqIHRoaXJkIHBhcnR5IGFuZCBzdWJqZWN0IHRvIGEgc2VwYXJhdGUgbGljZW5zZSwgYXZhaWxhYmxlXG4gKiBbaGVyZV0oaHR0cHM6Ly9naXRodWIuY29tL29wZW5haS9ncHQtMikuXG4gKlxuICpcbiAqIEV4YW1wbGUgdXNhZ2U6XG4gKiBgYGBqc1xuICogY29uc3QgdG9rZW5JZHMgPSB0Zi5vbmVzKFsxLCAxMl0pLCBkdHlwZT1cImludDMyXCIpO1xuICogY29uc3QgcGFkZGluZ01hc2sgPSB0Zi50ZW5zb3IoXG4gKiAgW1sxLCAxLCAxLCAxLCAxLCAxLCAxLCAxLCAxLCAxLCAwLCAwXV0sICdpbnQzMicpO1xuICpcbiAqICMgUHJldHJhaW5lZCBHUFQtMiBkZWNvZGVyLlxuICogbW9kZWwgPSBHUFQyQmFja2JvbmUuZnJvbVByZXNldChcImdwdDJfYmFzZV9lblwiKTtcbiAqIG1vZGVsLmFwcGx5KGlucHV0RGF0YSwge3BhZGRpbmdNYXNrfSk7XG4gKlxuICogIyBSYW5kb21seSBpbml0aWFsaXplZCBHUFQtMiBkZWNvZGVyIHdpdGggY3VzdG9tIGNvbmZpZy5cbiAqIG1vZGVsID0ga2VyYXNObHAubW9kZWxzLkdQVDJCYWNrYm9uZSh7XG4gKiAgICAgdm9jYWJ1bGFyeVNpemU6IDUwMjU3LFxuICogICAgIG51bUxheWVyczogMTIsXG4gKiAgICAgbnVtSGVhZHM6IDEyLFxuICogICAgIGhpZGRlbkRpbTogNzY4LFxuICogICAgIGludGVybWVkaWF0ZURpbTogMzA3MixcbiAqICAgICBtYXhTZXF1ZW5jZUxlbmd0aDogMTAyNCxcbiAqIH0pO1xuICogbW9kZWwuYXBwbHkoaW5wdXREYXRhLCB7cGFkZGluZ01hc2t9KTtcbiAqIGBgYFxuICovXG5leHBvcnQgY2xhc3MgR1BUMkJhY2tib25lIGV4dGVuZHMgQmFja2JvbmUge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGNsYXNzTmFtZSA9ICdHUFQyQmFja2JvbmUnO1xuXG4gIHByaXZhdGUgdm9jYWJ1bGFyeVNpemU6IG51bWJlcjtcbiAgcHJpdmF0ZSBudW1MYXllcnM6IG51bWJlcjtcbiAgcHJpdmF0ZSBudW1IZWFkczogbnVtYmVyO1xuICBwcml2YXRlIGhpZGRlbkRpbTogbnVtYmVyO1xuICBwcml2YXRlIGludGVybWVkaWF0ZURpbTogbnVtYmVyO1xuICBwcml2YXRlIGRyb3BvdXQ6IG51bWJlcjtcbiAgcHJpdmF0ZSBtYXhTZXF1ZW5jZUxlbmd0aDogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEdQVDJCYWNrYm9uZUFyZ3MpIHtcbiAgICBhcmdzLmRyb3BvdXQgPSBhcmdzLmRyb3BvdXQgPz8gMC4xO1xuICAgIGFyZ3MubWF4U2VxdWVuY2VMZW5ndGggPSBhcmdzLm1heFNlcXVlbmNlTGVuZ3RoID8/IDEwMjQ7XG5cbiAgICAvLyBJbnB1dHNcbiAgICBjb25zdCB0b2tlbklkcyA9IGlucHV0KHtzaGFwZTogW251bGxdLCBkdHlwZTogJ2ludDMyJywgbmFtZTogJ3Rva2VuX2lkcyd9KTtcbiAgICBjb25zdCBwYWRkaW5nTWFzayA9XG4gICAgICBpbnB1dCh7c2hhcGU6IFtudWxsXSwgZHR5cGU6ICdpbnQzMicsIG5hbWU6ICdwYWRkaW5nX21hc2snfSk7XG5cbiAgICAvLyBFbWJlZCB0b2tlbnMsIHBvc2l0aW9ucy5cbiAgICBjb25zdCB0b2tlbkVtYmVkZGluZyA9IG5ldyBFbWJlZGRpbmcoe1xuICAgICAgaW5wdXREaW06IGFyZ3Mudm9jYWJ1bGFyeVNpemUsXG4gICAgICBvdXRwdXREaW06IGFyZ3MuaGlkZGVuRGltLFxuICAgICAgZW1iZWRkaW5nc0luaXRpYWxpemVyOiBncHQyS2VybmVsSW5pdGlhbGl6ZXIoMC4wMSksXG4gICAgICBuYW1lOiAndG9rZW5fZW1iZWRkaW5nJyxcbiAgICB9KS5hcHBseSh0b2tlbklkcykgYXMgU3ltYm9saWNUZW5zb3I7XG5cbiAgICBjb25zdCBwb3NpdGlvbkVtYmVkZGluZyA9IG5ldyBQb3NpdGlvbkVtYmVkZGluZyh7XG4gICAgICBpbml0aWFsaXplcjogZ3B0Mktlcm5lbEluaXRpYWxpemVyKDAuMDIpLFxuICAgICAgc2VxdWVuY2VMZW5ndGg6IGFyZ3MubWF4U2VxdWVuY2VMZW5ndGgsXG4gICAgICBuYW1lOiAncG9zaXRpb25fZW1iZWRkaW5nJyxcbiAgICB9KS5hcHBseSh0b2tlbkVtYmVkZGluZykgYXMgU3ltYm9saWNUZW5zb3I7XG5cbiAgICAvLyBTdW0gYW5kIGFwcGx5IGRyb3BvdXQgdG8gZW1iZWRkaW5ncy5cbiAgICBsZXQgeCA9IGFkZCh7bmFtZTogJ2VtYmVkZGluZ3NfYWRkJ30pXG4gICAgICAuYXBwbHkoW3Rva2VuRW1iZWRkaW5nLCBwb3NpdGlvbkVtYmVkZGluZ10pIGFzIFN5bWJvbGljVGVuc29yO1xuICAgIHggPSBuZXcgRHJvcG91dCh7cmF0ZTogYXJncy5kcm9wb3V0LCBuYW1lOiAnZW1iZWRkaW5nc19kcm9wb3V0J30pXG4gICAgICAuYXBwbHkoeCkgYXMgU3ltYm9saWNUZW5zb3I7XG5cbiAgICAvLyBBcHBseSBzdWNjZXNzaXZlIHRyYW5zZm9ybWVyIGRlY29kZXIgYmxvY2tzLlxuICAgIGZvcihsZXQgaSA9IDA7IGkgPCBhcmdzLm51bUxheWVyczsgaSsrKSB7XG4gICAgICB4ID0gbmV3IFRyYW5zZm9ybWVyRGVjb2Rlcih7XG4gICAgICAgIGludGVybWVkaWF0ZURpbTogYXJncy5pbnRlcm1lZGlhdGVEaW0sXG4gICAgICAgIG51bUhlYWRzOiBhcmdzLm51bUhlYWRzLFxuICAgICAgICBkcm9wb3V0OiBhcmdzLmRyb3BvdXQsXG4gICAgICAgIGxheWVyTm9ybUVwc2lsb246IDFlLTA1LFxuICAgICAgICAvLyBUT0RPKHBmb3JkZXJpcXVlKTogSW1wbGVtZW50IGdlbHUuXG4gICAgICAgIGFjdGl2YXRpb246IGdldEFjdGl2YXRpb24oJ3JlbHUnKSxcbiAgICAgICAga2VybmVsSW5pdGlhbGl6ZXI6IGdwdDJLZXJuZWxJbml0aWFsaXplcigwLjAyKSxcbiAgICAgICAgbm9ybWFsaXplRmlyc3Q6IHRydWUsXG4gICAgICAgIG5hbWU6IGB0cmFuc2Zvcm1lcl9sYXllcl8ke2l9YCxcbiAgICAgIH0pLmFwcGx5KHgsIHtkZWNvZGVyUGFkZGluZ01hc2s6IHBhZGRpbmdNYXNrfSkgYXMgU3ltYm9saWNUZW5zb3I7XG4gICAgfVxuXG4gICAgY29uc3Qgc2VxdWVuY2VPdXRwdXQgPSBuZXcgTGF5ZXJOb3JtYWxpemF0aW9uKHtcbiAgICAgIG5hbWU6ICdsYXllcl9ub3JtJyxcbiAgICAgIGF4aXM6IC0xLFxuICAgICAgZXBzaWxvbjogMWUtMDUsXG4gICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgIH0pLmFwcGx5KHgpIGFzIFN5bWJvbGljVGVuc29yO1xuXG4gICAgLy8gSW5zdGFudGlhdGUgdXNpbmcgRnVuY3Rpb25hbCBBUEkgTW9kZWwgY29uc3RydWN0b3IuXG4gICAgc3VwZXIoe1xuICAgICAgaW5wdXRzOiBbdG9rZW5JZHMsIHBhZGRpbmdNYXNrXSxcbiAgICAgIG91dHB1dHM6IHNlcXVlbmNlT3V0cHV0LFxuICAgICAgbmFtZTogJ2dwdDJfYmFja2JvbmUnXG4gICAgfSk7XG4gICAgdGhpcy52b2NhYnVsYXJ5U2l6ZSA9IGFyZ3Mudm9jYWJ1bGFyeVNpemU7XG4gICAgdGhpcy5udW1MYXllcnMgPSBhcmdzLm51bUxheWVycztcbiAgICB0aGlzLm51bUhlYWRzID0gYXJncy5udW1IZWFkcztcbiAgICB0aGlzLmhpZGRlbkRpbSA9IGFyZ3MuaGlkZGVuRGltO1xuICAgIHRoaXMuaW50ZXJtZWRpYXRlRGltID0gYXJncy5pbnRlcm1lZGlhdGVEaW07XG4gICAgdGhpcy5kcm9wb3V0ID0gYXJncy5kcm9wb3V0ID8/IDAuMTtcbiAgICB0aGlzLm1heFNlcXVlbmNlTGVuZ3RoID0gYXJncy5tYXhTZXF1ZW5jZUxlbmd0aCA/PyAxMDI0O1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICB2b2NhYnVsYXJ5U2l6ZTogdGhpcy52b2NhYnVsYXJ5U2l6ZSxcbiAgICAgIG51bUxheWVyczogdGhpcy5udW1MYXllcnMsXG4gICAgICBudW1IZWFkczogdGhpcy5udW1IZWFkcyxcbiAgICAgIGhpZGRlbkRpbTogdGhpcy5oaWRkZW5EaW0sXG4gICAgICBpbnRlcm1lZGlhdGVEaW06IHRoaXMuaW50ZXJtZWRpYXRlRGltLFxuICAgICAgZHJvcG91dDogdGhpcy5kcm9wb3V0LFxuICAgICAgbWF4U2VxdWVuY2VMZW5ndGg6IHRoaXMubWF4U2VxdWVuY2VMZW5ndGgsXG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdG9rZW5FbWJlZGRpbmcoKTogRW1iZWRkaW5nIHtcbiAgICByZXR1cm4gdGhpcy5nZXRMYXllcigndG9rZW5fZW1iZWRkaW5nJykgYXMgRW1iZWRkaW5nO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR1BUMkJhY2tib25lKTtcbiJdfQ==