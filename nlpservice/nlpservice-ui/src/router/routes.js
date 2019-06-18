import Start from '../components/Start'
import Summarize from '../components/summarize/Main'
import Duplicate from '../components/duplicate/Main'
import Similarity from '../components/Similarity'
import Classify from '../components/Classify'
import Clusterize from '../components/cluster/Main'

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
    path: '/duplicate',
    component: Duplicate,
  },
  {
    path: '/classify',
    component: Classify,
  },
]

export default routes
