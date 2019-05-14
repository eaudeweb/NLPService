import Start from '../components/Start'
import Summarize from '../components/summarize/Main'
import Duplicate from '../components/duplicate/Main'
import Similarity from '../components/Similarity'

const routes = [
  {
    path: '/',
    component: Start,
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
]

export default routes
