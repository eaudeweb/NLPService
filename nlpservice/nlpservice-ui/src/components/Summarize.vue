<template>
  <v-container>
    <v-layout text-xs-center wrap>

      <v-flex mb-5 xs12>
        <h2 class="headline font-weight-bold mb-3">Summarize</h2>
        <v-sheet color="#DDD" elevation="1" min-height="10em">
          {{ summary }}
        </v-sheet>

        <v-form>
          <v-textarea v-model="text"></v-textarea>
          <v-btn @click="submit">Submit</v-btn>
        </v-form>

        <v-layout justify-center></v-layout>
      </v-flex>

    </v-layout>
  </v-container>
</template>
<script>
  import axios from 'axios'

  export default {
    data () {
      return {
        'summary': '',
        'text': '',
      }
    },
    methods: {
      submit() {
        axios
          .post('http://localhost:6543/summarize', {'text': this.text})
          .then((resp) => {
            this.summary = resp.data.summary.preview
            console.log(resp)
          })
      }
    }
  }
</script>
<style>
</style>
