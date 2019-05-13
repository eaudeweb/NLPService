<template>
  <v-container grid-list-md>
    <v-layout row wrap>

      <v-flex xs12 md6 item>
        <v-sheet theme="light">
          <h2 class="headline font-weight-bold mb-3">Summary</h2>
          <v-sheet color="#DDD" elevation="1" min-height="10em">
            {{ summary }}
          </v-sheet>
          <v-btn>Copy to clibboard</v-btn>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md6 item>
        <v-container grid-list-md>
          <v-layout column wrap>
            <v-flex>
              <v-card tile flat color="amber lighten-2">
                <v-card-text>
                  <v-form>
                    Enter text:
                    <v-textarea v-model="text"></v-textarea>
                  </v-form>
                </v-card-text>
                <v-card-actions>
                  <v-btn flat @click="submit">Submit</v-btn>
                </v-card-actions>
              </v-card>
            </v-flex>
            <v-flex>
              <v-card tile flat color="purple lighten-4">
                <v-card-text>
                  <div v-for="sent in sentences">
                    {{ sent }}
                  </div>
                </v-card-text>
              </v-card>
            </v-flex>
          </v-layout>
        </v-container>

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
        'sentences': [],
      }
    },
    methods: {
      submit() {
        axios
          .post('http://localhost:6543/summarize', {'text': this.text})
          .then((resp) => {
            let data = resp.data.summary
            this.summary = data.preview
            this.sentences = data.sentences
            console.log(resp)
          })
      }
    }
  }
</script>
<style>
</style>
