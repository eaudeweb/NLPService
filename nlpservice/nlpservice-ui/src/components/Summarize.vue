<template>
  <v-container grid-list-md>
    <v-layout row wrap>
      <v-flex xs12 md12>
      <h2 class="headline font-weight-bold mb-3">Summary</h2>
      </v-flex>

      <v-flex xs12 md6>
        <v-sheet theme="light">
          <v-sheet color="#DDD" elevation="1" min-height="10em">
            {{ summary }}
          </v-sheet>
          <v-btn>Copy to clibboard</v-btn>
        </v-sheet>
      </v-flex>

      <v-flex xs12 md6>
          <v-layout column wrap>
            <v-flex>
              <v-card tile flat color="deep-purple lighten-5">
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
              <v-card tile flat>
                <v-list>
                  <v-list-tile v-for="(clust, ix) in clusters">
                    <v-sheet :color="getClusterColor(ix)" class="mb-1">
                      <div v-for="sent in clust">
                        <sentence :sent="sent"></sentence>
                      </div>
                    </v-sheet>
                  </v-list-tile>
                </v-list>
              </v-card>
            </v-flex>
          </v-layout>

      </v-flex>

    </v-layout>
  </v-container>
</template>
<script>
// [ { "text": "The attack was developed by Israeli security firm NSO Group,
//   according to a report in the Financial Times.", "is_summary": false, "keep":
//   false }, { "text": "The attack was first discovered earlier this month.",
//     "is_summary": false, "keep": false }, { "text": "It involved attackers
//       using WhatsApp's voice calling function to ring a target's device.",
//       "is_summary": true, "keep": false }, { "text": "Even if the call was not
//         picked up, the surveillance software would be installed, and, the FT
//         reported, the call would often disappear from the device's call log.",
//         "is_summary": false, "keep": false } ]
  import axios from 'axios'
  import colors from 'vuetify/es5/util/colors'
  import Sentence from './Sentence'

  export default {
    components: {
      Sentence,
    },
    data () {
      return {
        'summary': '',
        'text': '',
        'clusters': [],
      }
    },
    methods: {
      getClusterColor(ix) {
        let blacklist = ['lightBlue', 'deepPurple']
        // let colIx = Math.floor(Math.random() * colNames.length)
        let colNames = Object.keys(colors).filter(n => blacklist.indexOf(n) == -1)
        let colIx = ix % colNames.length
        return colNames[colIx] + ' lighten-4'
      },
      submit() {
        axios
          .post('http://localhost:6543/summarize', {'text': this.text})
          .then((resp) => {
            let data = resp.data.summary
            this.summary = data.preview
            this.clusters = data.sentences
            console.log(resp)
          })
      }
    }
  }
</script>
<style>
</style>
