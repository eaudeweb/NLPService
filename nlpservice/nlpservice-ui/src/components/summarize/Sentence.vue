<template>
  <div :class="cssClass">
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
