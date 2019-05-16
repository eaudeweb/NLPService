<template>
  <v-container grid-list-md>
    <v-layout row wrap>
      <v-flex xs12 md12>
        <h2 class="headline font-weight-bold mb-3">Sentence clusterization</h2>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="green lighten-3" elevation="1" min-height="10em">
          <h4>Enter text (one sentence per line)</h4>
          <v-textarea v-model="text"></v-textarea>
        </v-sheet>
        <v-sheet color="blue lighten-3" elevation="1" min-height="10em">
          <h4>Enter topic terms</h4>
          <p>
          One topic per line, each topic can have multiple comma-separated
          concept words
          </p>
          <v-textarea v-model="topics"></v-textarea>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="purple lighten-3" elevation="1" min-height="10em">
          <h4>Topic clusters</h4>
          <template v-for="sent in sentences">
            <sentence :sent="sent"></sentence>
          </template>
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
import Sentence from './Sentence'

export default {
  components: {
    Sentence,
  },
  data () {
    return {
      'text': '',
      'topics': '',
      'sentences': [],
    }
  },
  methods: {
    submit() {
      console.log(this.topics)
      // console.log(this.text)
      axios
        .post('http://localhost:6543/clusterize', {
            'text': this.text,
            'topics': this.topics,
          })
        .then((resp) => {
          console.log(resp)
          this.sentences = resp.data.result
        })
    }
  }
}
</script>
<style>
</style>
