import {createRouter, createWebHistory} from 'vue-router'


const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'login',
      component: () => import('@/views/Login.vue'),
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('@/views/Register.vue'),
    },
    {
      path: '/flower',
      name: 'flower',
      component: () => import('@/views/Flower.vue'),
    },
    {
      path: '/message',
      name: 'message',
      component: () => import('@/views/message.vue'),
    },
    {
      path: '/camera',
      name: 'camera',
      component: () => import('@/views/Camera.vue'),
    }
  ]
})

export default router
