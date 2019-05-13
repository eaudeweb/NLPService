<template>
  <v-app>
    <v-toolbar app>
      <v-toolbar-title class="headline text-uppercase">
        <span>NLService</span>
        <span class="font-weight-light">quick demo</span>
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn
        flat
        href="https://eaudeweb.ro"
        target="_blank"
        >
        <span class="mr-2">NLP Service</span>
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
