import Start from '../components/Start'
import Summarize from '../components/Summarize'
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
]

export default routes
