<template>
  <div class="publish">
    <video ref="video"></video>
    <canvas style="display: none" id="canvasCamera"></canvas>
    <div v-if="imgSrc" class="img_bg_camera">
      <img :src="imgSrc" class="tx_img" />
    </div>
    <button @click="back">返回</button>
    <button @click="openCamera">打开摄像头</button>
    <button @click="closeCamera">关闭摄像头</button>
    <button @click="takePhoto">拍照</button>
    <button v-if="showConfirmButton" @click="confirmPhoto">确定</button>
  </div>
</template>

<script>
import router from "../router";

export default {
  data() {
    return {
      mediaStreamTrack: {},
      video_stream: '', // 视频stream
      imgSrc: '', // 拍照图片
      canvas: null,
      context: null,
      showConfirmButton: false, // 控制确定按钮显示与隐藏
    };
  },
  mounted() {
    // 进入页面 自动调用摄像头
    this.getCamera();
  },
  methods: {
    // 调用打开摄像头功能
    getCamera() {
      // 获取 canvas 画布
      this.canvas = document.getElementById('canvasCamera');
      this.context = this.canvas.getContext('2d');
      // 旧版本浏览器可能根本不支持mediaDevices，我们首先设置一个空对象
      if (navigator.mediaDevices === undefined) {
        navigator.mediaDevices = {};
      }
      // 正常支持版本
      navigator.mediaDevices
          .getUserMedia({
            video: true,
          })
          .then((stream) => {
            // 摄像头开启成功
            this.mediaStreamTrack = typeof stream.stop === 'function' ? stream : stream.getTracks()[0];
            this.video_stream = stream;
            this.$refs.video.srcObject = stream;
            this.$refs.video.play();
          })
          .catch(err => {
            console.log(err);
          });
    },
    // 拍照 绘制图片
    takePhoto() {
      console.log('拍照');
      // 点击canvas画图
      this.context.drawImage(
          this.$refs.video,
          0,
          0,
          200,
          100,
      );
      // 获取图片base64链接
      const image = this.canvas.toDataURL('image/png');
      this.imgSrc = image;
      this.showConfirmButton = true; // 显示确定按钮
    },
    // 确定按钮点击事件
    confirmPhoto() {
      // 将拍摄的照片传递到花卉识别页面并导航
      localStorage.setItem('student-camera', JSON.stringify(this.imgSrc))
      console.log(this.imgSrc)
      router.push('/flower')
    },
    back(){
      router.push('/flower')
    },
    // 打开摄像头
    openCamera() {
      console.log('打开摄像头');
      this.getCamera();
    },
    // 关闭摄像头
    closeCamera() {
      console.log('关闭摄像头');
      this.$refs.video.srcObject.getTracks()[0].stop();
    },
  },
};
</script>

<style scoped>
video {
  width: 100%;
  height: 300px;
}
canvas {
  width: 100%;
  height: 300px;
}
button {
  width: 100px;
  height: 40px;
  position: relative;
  bottom: 0;
  left: 0;
  background-color: rgb(22, 204, 195);
}
.img_bg_camera img {
  width: 300px;
  height: 200px;
}
</style>
