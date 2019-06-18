<template>
  <v-container grid-list-md>
    <v-layout row wrap>
      <v-flex xs12 md12>
      <h2 class="headline font-weight-bold mb-3">Similarity</h2>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="green lighten-3" elevation="1" min-height="10em">
          <h4>Enter text</h4>
          <v-textarea v-model="text"></v-textarea>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="purple lighten-3" elevation="1" min-height="10em">
          <div v-for="s in scores">
            {{ s[0] }} - {{ s[1] }}
          </div>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md12>
        <v-btn primary @click="submit">Submit</v-btn>
      </v-flex>

    </v-layout>
  </v-container>
</template>
<script>
  import axios from 'axios'

  export default {
    components: {
    },
    data () {
      return {
        'text': '',
        'scores': [],
      }
    },
    methods: {
      submit() {
        axios
          .post('http://localhost:6543/classify',
            {
              'text': this.text,
              'model': 'kg',
            })
          .then((resp) => {
            this.scores = resp.data.result
            console.log(resp)
          })
      }
    }
  }
</script>
<style>
</style>
