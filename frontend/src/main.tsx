// import React from 'react'
// import ReactDOM from 'react-dom/client'
// import { BrowserRouter } from 'react-router-dom'
// import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// import App from './App'
// import './index.css'

// const queryClient = new QueryClient({
//   defaultOptions: { queries: { retry: 1, staleTime: 30000 } }
// })

// ReactDOM.createRoot(document.getElementById('root')!).render(
//   <React.StrictMode>
//     <QueryClientProvider client={queryClient}>
//       <BrowserRouter>
//         <App />
//       </BrowserRouter>
//     </QueryClientProvider>
//   </React.StrictMode>
// )

import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      // staleTime: 0 means cached data is immediately considered stale.
      // React Query will always do a background refetch when a component mounts.
      // Previously this was 30000 (30s) which caused documents to not reload
      // after login because the old empty cache was still "fresh".
      staleTime: 0,
      // Always refetch when the component that uses the query re-mounts.
      // This ensures that after login → logout → login, queries run fresh.
      refetchOnMount: true,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
)
