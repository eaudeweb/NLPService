import Classify from '../components/Classify'
import Clusterize from '../components/cluster/Main'
import Duplicate from '../components/duplicate/Main'
import KeyedVectors from '../components/KeyedVectors'
import Retrain from '../components/Retrain'
import Similarity from '../components/Similarity'
import Summarize from '../components/summarize/Main'
import Start from '../components/Start'

const routes = [
  {
    path: '/',
    component: Start,
  },
  {
    path: '/clusterize',
    component: Clusterize,
  },
  {
    path: '/summarize',
    component: Summarize,
  },
  {
    path: '/similarity',
    component: Similarity,
  },
  {
    path: '/kv',
    component: KeyedVectors,
  },
  {
    path: '/duplicate',
    component: Duplicate,
  },
  {
    path: '/classify',
    component: Classify,
  },
  {
    path: '/classify/retrain',
    component: Retrain,
  },
]

export default routes
