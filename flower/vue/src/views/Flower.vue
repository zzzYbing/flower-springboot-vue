<template>
  <div class="login-container">
    <div style="width: 400px" class="flower-box">
      <div class="card" style="line-height: 30px; display: flex; justify-content: space-between;">
        <div>
          欢迎您，<span style="color: dodgerblue;">{{ user.name }}</span>
        </div>
        <div>
          <button @click="back">退出登录</button>
        </div>
      </div>
      <h1>花卉识别系统</h1>
      <br/>
      <div class="tab_title">
        <ul>
          <li @click="tab=0" :class="{tab_active:tab==0}">
            查询
          </li>
          <li @click="tab=1" :class="{tab_active:tab==1}">
            识别
          </li>
        </ul>
      </div>
      <div class="content">
        <div class="tab_content" v-if="tab==1">
          <div class="button-container">
            <label for="file-upload" class="custom-file-upload" :disabled="!modelLoaded">
              选择文件
            </label>
            <button @click="poh" class="take-photo-button">拍照上传</button>
          </div>
          <input id="file-upload" type="file" @change="uploadImage" accept="image/*" multiple style="display: none;"/>
          <br/>
          <div>
            <div class="image-container">
              <img :src="imageUrl" id="photo" :alt="altText"/>
            </div>
          </div>
          <h3 id="status">状态：{{ status }}</h3>
          <button @click="analyze" :disabled="!modelLoaded">点击识别</button>
          <div v-if="result !== ''">
            <h3>识别结果：</h3>
            <p>{{ result }}</p>
            <button @click="nextpage"> 查看养护信息</button>
          </div>
        </div>
        <div class="tab_content" v-if="tab==0">
          <input id="put" class="search-input" placeholder="请输入花卉种类名称"/>
          <button @click="search">搜 索</button>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  data() {
    return {
      tab: 0 //默认选中第一个tab
    };
  }
};
</script>
<script setup>
import * as tf from '@tensorflow/tfjs';
import request from "../utils/request";


const user = JSON.parse(localStorage.getItem('student-user') || '{}');
const camera = JSON.parse(localStorage.getItem('student-camera') || '{}');

let model = null;
let imageUrl = ref(camera);
let altText = ref('还没有上传图片，请点击选择文件按钮上传图片，然后按识别按键开始识别。');
let status = ref('无');
let result = ref('');
const IMAGENET_CLASSES = ['小雏菊', '蒲公英', '玫瑰花', '向日葵', '郁金香'];


import {ref} from 'vue';
import {ElMessage} from "element-plus";
import router from "../router";
import {string} from "@tensorflow/tfjs";

const modelLoaded = ref(false);

const formRef = ref()

async function search() {
  if (put.value !== '小雏菊' && put.value !== '蒲公英' && put.value !== '玫瑰花' && put.value !== '向日葵' && put.value !== '郁金香') {
    ElMessage.error('没有搜索到该花卉！');
  } else {
    request.post('/flower', {name: put.value}).then(res => {
      if (res.code === '200') {
        console.log("-------------------");
        localStorage.setItem('student-flower', JSON.stringify(res.data));
        console.log(res.data);
        ElMessage.success('成功');
        request.post('/comments', {category: put.value}).then(res => {
          if (res.code === '200') {
            localStorage.setItem('student-comments', JSON.stringify(res.data));
            console.log("-------------------");
            console.log(res.data);
            ElMessage.success('成功');
            router.push('/message');
          } else {
            ElMessage.error(res.masg);
          }
        });
      } else {
        ElMessage.error(res.msg);
      }
    });
  }
}

loadModel();

async function loadModel() {
  status.value = '模型加载中，请稍后...';
  console.log('模型加载中...');

  model = await tf.loadLayersModel('./ModelJS/model.json');
  status.value = '模型加载完成!';
  modelLoaded.value = true;

  console.log('模型加载完成...');
}

async function back() {
  router.push('/')

}

async function poh() {
  router.push('/camera')

}

async function uploadImage(event) {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    imageUrl.value = reader.result;
    localStorage.setItem('student-camera', JSON.stringify(imageUrl.value))
    console.log(imageUrl)
    console.log(imageUrl.value)
  };
}

async function analyze() {
  status.value = '识别中，请稍后...';

  const imgElement = document.getElementById('photo');
  const img = tf.browser.fromPixels(imgElement).resizeBilinear([224, 224]).toFloat().expandDims();

  const logits = model.predict(img);
  const predictions = await logits.data();
  const topClassIndices = tf.argMax(predictions).dataSync(); // 获取最高概率的类别索引
  const topClassProbability = predictions[topClassIndices]; // 获取最高概率
  result.value = `${IMAGENET_CLASSES[topClassIndices]}: ${(topClassProbability * 100).toFixed(2)}%`; // 显示识别结果
  console.log(result);
  console.log(result.value);
  console.log(IMAGENET_CLASSES[topClassIndices]);
  status.value = '识别完成!';
  const name = IMAGENET_CLASSES[tf.argMax(await model.predict(tf.browser.fromPixels(document.getElementById('photo')).resizeBilinear([224, 224]).toFloat().expandDims()).data()).dataSync()];
  const category = name;
  request.post('/comments', {category: category}).then(res => {
    if (res.code === '200') {
      localStorage.setItem('student-comments', JSON.stringify(res.data))
      console.log("-------------------")
      console.log(res.data)
      ElMessage.success('成功')
    } else {
      ElMessage.error(res.masg)
    }
  })
}

async function nextpage() {
  const name = IMAGENET_CLASSES[tf.argMax(await model.predict(tf.browser.fromPixels(document.getElementById('photo')).resizeBilinear([224, 224]).toFloat().expandDims()).data()).dataSync()];
  const category = name;

  request.post('/flower', {name: name}).then(res => {
    if (res.code === '200') {
      console.log("-------------------")
      localStorage.setItem('student-flower', JSON.stringify(res.data))
      console.log(res.data)
      ElMessage.success('成功')
      router.push('/message')
    } else {
      ElMessage.error(res.msg)
    }
  })
}
</script>

<style>
.tab_title {
  margin-bottom: 20px; /* 添加间距，使布局更清晰 */
}

.tab_title ul {
  list-style-type: none; /* 移除列表项默认样式 */
  padding: 0; /* 移除内边距 */
  margin: 0; /* 移除外边距 */
  display: flex; /* 设置为弹性布局 */
  justify-content: space-between; /* 平均分配子项间距 */
}

.tab_title li {
  cursor: pointer; /* 设置鼠标指针为手型 */
  padding: 10px 20px; /* 添加内边距，增加可点击区域 */
  border: 1px solid #ccc; /* 添加边框 */
  border-radius: 5px; /* 圆角边框 */
  flex-grow: 1; /* 均匀分配子项宽度 */
  text-align: center; /* 居中文字 */
}

.tab_title li.tab_active {
  background-color: #007bff; /* 设置选中项背景颜色 */
  color: #fff; /* 设置选中项文字颜色 */
}

.search-container {
  display: flex;
  align-items: center;
}

.search-input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin-right: 8px;
}

.login-container {
  min-height: 100vh;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background-image: url("../assets/imgs/flower.jpg");
  background-size: cover;
}

.flower-box {
  position: absolute; /* 使用绝对定位 */
  top: 30px; /* 与页面顶部对齐 */
  background-color: rgba(255, 255, 255, .8);
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  padding: 30px;
  border-radius: 5px;
}

button {
  background-color: dodgerblue; /* 按钮背景颜色 */
  color: white; /* 按钮文字颜色 */
  border: none; /* 移除按钮边框 */
  padding: 10px 20px; /* 按钮内边距 */
  border-radius: 5px; /* 按钮圆角 */
  cursor: pointer; /* 鼠标悬停样式为手型 */
  transition: background-color 0.3s, color 0.3s; /* 添加过渡效果 */
}

button:hover {
  background-color: #007acc; /* 按钮悬停时的背景颜色 */
  color: #f0f0f0; /* 按钮悬停时的文字颜色 */
}
.button-container {
  display: flex; /* 使用 flex 布局 */
}
.custom-file-upload {
  background-color: dodgerblue; /* 按钮背景颜色 */
  color: white; /* 按钮文字颜色 */
  border: none; /* 移除按钮边框 */
  padding: 10px 20px; /* 按钮内边距 */
  border-radius: 5px; /* 按钮圆角 */
  cursor: pointer; /* 鼠标悬停样式为手型 */
  transition: background-color 0.3s, color 0.3s; /* 添加过渡效果 */
}

.custom-file-upload:hover {
  background-color: #007acc; /* 按钮悬停时的背景颜色 */
  color: #f0f0f0; /* 按钮悬停时的文字颜色 */
}

.image-container {
  height: 250px; /* 在这里设置你想要的固定高度 */
  overflow: hidden; /* 隐藏任何溢出内容 */
}

.image-container img {
  width: auto; /* 保持纵横比 */
  height: 100%; /* 填充容器的高度 */
}
</style>