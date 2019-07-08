<template>
  <v-container grid-list-md>
    <v-layout row wrap>
      <v-flex xs12 md12>
      <h2 class="headline font-weight-bold mb-3">Retrain</h2>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet color="green lighten-3" elevation="1" min-height="10em">
          <h4>Set models for retraining</h4>
          <v-select multiple :items="models" label="Model" @change="setModel"></v-select>
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
        'models': [],
        'retrain': []
      }
    },
    methods: {
      setModel(value) {
        this.retrain = value
      },
      submit() {
        axios
          .post('http://localhost:6543/classify/',
            {
              'models': this.retrain,
            })
          .then((resp) => {
            this.scores = resp.data.result
          })
      }
    },
    mounted() {
        axios
          .get('http://localhost:6543/classify/')
          .then((resp) => {
            this.models = resp.data.result
          })
    }
  }
</script>
<style>
</style>
