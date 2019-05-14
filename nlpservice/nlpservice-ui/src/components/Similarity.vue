<template>
  <v-container grid-list-md>
    <v-layout row wrap>
      <v-flex xs12 md12>
      <h2 class="headline font-weight-bold mb-3">Similarity</h2>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="green lighten-3" elevation="1" min-height="10em">
          <h4>Base text (one sentence per line)</h4>
          <v-textarea v-model="base"></v-textarea>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="purple lighten-3" elevation="1" min-height="10em">
          <h4>Proba text (one sentence per line)</h4>
          <v-textarea v-model="proba"></v-textarea>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md12>
        <v-btn primary @click="submit">Submit</v-btn>
        <h2 v-if="score" class='text-align-center'>{{score}}</h2>
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
        'base': '',
        'proba': '',
        'score': null,
      }
    },
    methods: {
      submit() {
        axios
          .post('http://localhost:6543/similarity',
            {
              'base': this.base,
              'proba': this.proba,
            })
          .then((resp) => {
            this.score = resp.data.score
            console.log(resp)
          })
      }
    }
  }
</script>
<style>
</style>

