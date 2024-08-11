<template>
  <div class="page-background">
    <div class="flower-details">
      <button @click="back" >返回</button>
      <h2>{{ flower.name }}</h2>
      <div class="video-container">
        <video
            class="video-player"
            :src="videoSrc"
            controls
            ref="videoRef"
        ></video>
      </div>
      <div class="details">
        <div class="detail-item">
          <strong>形态特征:</strong>
          <p>{{ flower.morphological }}</p>
        </div>
        <div class="detail-item">
          <strong>生长习性:</strong>
          <p>{{ flower.habits }}</p>
        </div>
        <div class="detail-item">
          <strong>光照:</strong>
          <p>{{ flower.illumination }}</p>
        </div>
        <div class="detail-item">
          <strong>温度:</strong>
          <p>{{ flower.temperature }}</p>
        </div>
        <div class="detail-item">
          <strong>水分:</strong>
          <p>{{ flower.watering }}</p>
        </div>
        <div class="detail-item">
          <strong>施肥:</strong>
          <p>{{ flower.fertilize }}</p>
        </div>
      </div>
      <div>
        <form @submit.prevent="submitComment" class="comment-form">
          <textarea id="comment" v-model="newCommentContent" class="comment-textarea"></textarea><br/>
          <button type="submit" class="submit-button">提交评论</button>
        </form>
      </div>
      <div>
        <h3>评论：</h3>
        <ul>
          <li v-for="comment in comments" :key="comment.id">
            <p><strong>{{ comment.user }}</strong>:</p>
            <p>{{ comment.content }}</p>
            <p>{{ comment.time }}</p>
          </li>
        </ul>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import router from "../router";
import request from "../utils/request";
import {ElMessage} from "element-plus";


const newCommentContent = ref('');
const comments = ref(JSON.parse(localStorage.getItem('student-comments') || '{}'));
const flower = JSON.parse(localStorage.getItem('student-flower') || '{}');
const admin = JSON.parse(localStorage.getItem('student-user') || '{}');



const videoRef = ref(null); // 创建一个响应式引用

let videoSrc = "/Video/daisy.mp4"; // 默认视频路径为雏菊视频

if (flower.name === '玫瑰花') {
  videoSrc = "/Video/rose.mp4";
} else if (flower.name === '郁金香') {
  videoSrc = "/Video/tulips.mp4";
} else if (flower.name === '向日葵') {
  videoSrc = "/Video/sunflower.mp4";
} else if (flower.name === '蒲公英') {
  videoSrc = "/Video/dandelion.mp4";
}
async function back(){
  router.push('/flower')

}
onMounted(() => {
  const video = videoRef.value;
  if (video) {
    video.src = videoSrc; // 设置视频源
  }
});


async function submitComment() {
  const commentData = {
    content: newCommentContent.value,
    category: flower.name,
    user: admin.name
  };

  try {
    const addCommentResponse = await request.post('/addcomment', commentData);
    if (addCommentResponse.code === '200') {
      ElMessage.success('评论成功');
      newCommentContent.value = "";

      // 重新获取评论数据并更新页面
      const getCommentsResponse = await request.post('/comments', { category: flower.name });
      if (getCommentsResponse.code === '200') {
        localStorage.setItem('student-comments', JSON.stringify(getCommentsResponse.data));
        comments.value = getCommentsResponse.data; // 更新评论数据
      } else {
        ElMessage.error(getCommentsResponse.msg);
      }
    } else {
      ElMessage.error(addCommentResponse.msg);
    }
  } catch (error) {
    console.error('Error submitting comment:', error);
  }
}

</script>

<style scoped>
.page-background {
  background-image: url('../assets/imgs/message.jpg');
  background-size: cover;
  background-position: center;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

.flower-details {
  background-color: rgba(255, 255, 255, 0.8);
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}

h2 {
  margin-bottom: 20px;
  color: #333333;
}

.details {
  display: flex;
  flex-direction: column; /* 修改为竖直排列 */
}

.detail-item {
  background-color: #f5f5f5;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 10px; /* 添加间距 */
}

.detail-item strong {
  display: block;
  font-weight: bold;
  margin-bottom: 10px;
}

.detail-item p {
  margin: 0;
}

.video-container {
  width: 100%;
  max-width: 800px; /* 设置最大宽度，防止视频过宽 */
  margin: 0 auto; /* 水平居中 */
}

.video-player {
  width: 100%; /* 视频宽度占满父容器 */
}
@media screen and (max-width: 767px) {
  .video-player:-webkit-full-screen {
    transform: rotate(90deg) translateY(-50%);
    width: 100vh;
    height: 100vw;
    margin-left: 50%;
    margin-top: 50%;
  }
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

.comment-form {
  width: 100%;
}

.comment-textarea {
  width: calc(100% - 20px); /* 20px用于留出一些空间以防止溢出 */
  height: 100px; /* 设置合适的高度 */
  padding: 10px; /* 设置内边距 */
  font-size: 16px; /* 设置字体大小 */
  border: 1px solid #ccc; /* 设置边框 */
}

.submit-button {
  margin-top: 10px; /* 设置按钮顶部边距 */
  padding: 8px 16px; /* 设置按钮内边距 */
  font-size: 16px; /* 设置按钮字体大小 */
}

</style>
