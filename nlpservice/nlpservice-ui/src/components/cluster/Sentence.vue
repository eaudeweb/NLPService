<template>
  <div :class="cssClass">
    <v-tooltip bottom>
      <template v-slot:activator="{ on }">
        <span v-on="on">{{ sent.topic }}</span>
      </template>
      <span>{{ sent.seed }}</span>
    </v-tooltip>
    {{ sent.text }}
  </div>
</template>
<script>
  import colors from 'vuetify/es5/util/colors'
export default {
  props: ['sent'],
  methods: {},
  computed: {
    cssClass () {
      let blacklist = ['lightBlue', 'deepPurple']
      let colNames = Object.keys(colors).filter(n => blacklist.indexOf(n) == -1)
      let colIx = this.sent.topic % colNames.length
      console.log(this.sent.topic, colIx)
      return colNames[colIx] + ' lighten-4'
    }
  }

}
</script>
