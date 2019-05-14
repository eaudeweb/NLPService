<template>
  <div :class="cssClass">
    <v-tooltip bottom>
      <template v-slot:activator="{ on }">
        <span v-on="on">{{ sent.c }}</span>
      </template>
      <span>{{ sent.score }}</span>
    </v-tooltip>
    {{ sent.text }}
  </div>
</template>
<script>
  export default {
    props: ['sent'],
    methods: {
    },
    computed: {
      cssClass () {
        console.log(this.sent)
        let klass = 'pa-1 '
        if (this.sent.is_summary) {
          klass += ' red--text'
        }
        if (this.sent.keep) {
          klass += ' darken-2'
        }
        return klass
      }
    }

  }
</script>

