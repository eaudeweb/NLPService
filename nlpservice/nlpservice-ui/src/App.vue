<template>
  <v-app>
    <v-toolbar app>
      <v-toolbar-title class="headline text-uppercase">
        <span>NLService</span>
        <span class="font-weight-light">quick demo</span>
      </v-toolbar-title>
      <v-spacer></v-spacer>

      <v-btn flat>
        <router-link to="/clusterize">Clusterize</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/duplicate">Duplicate</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/similarity">Similarity</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/kv">KV</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/summarize">Summarize</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/classify">Classify</router-link>
      </v-btn>
      <v-btn flat>
        <router-link to="/classify/retrain">Retrain</router-link>
      </v-btn>

    </v-toolbar>

    <v-content>
      <router-view></router-view>
    </v-content>

  </v-app>
</template>

<script>

  export default {
    name: 'App',
    components: {},
    data () {
      return {}
    }
  }
</script>
