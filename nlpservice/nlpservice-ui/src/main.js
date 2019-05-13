import Vue from 'vue'
import './plugins/vuetify'
import App from './App.vue'
import routes from './router/routes.js'
import VueRouter from 'vue-router'

Vue.config.productionTip = false
Vue.use(VueRouter);

const router = new VueRouter({
  routes    // short for `routes: routes`
})

new Vue({
  router,
  render: h => h(App),
}).$mount('#app')
