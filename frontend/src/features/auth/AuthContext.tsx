import React, { createContext, useContext, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

interface AuthUser { id: number; email: string; token: string }
interface AuthContextType {
  user: AuthUser | null
  login: (token: string, id: number, email: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType>(null!)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const token = localStorage.getItem('token')
    const id = localStorage.getItem('userId')
    const email = localStorage.getItem('userEmail')
    if (token && id && email) return { token, id: parseInt(id), email }
    return null
  })

  // useQueryClient is available here because AuthProvider is rendered
  // inside QueryClientProvider in main.tsx
  const queryClient = useQueryClient()

  const login = (token: string, id: number, email: string) => {
    localStorage.setItem('token', token)
    localStorage.setItem('userId', String(id))
    localStorage.setItem('userEmail', email)
    setUser({ token, id, email })

    // Invalidate all cached queries so they refetch with the new user's token.
    // Without this, React Query may serve stale cached data (e.g. empty document
    // list from a previous session) because staleTime: 30000 prevents refetching.
    queryClient.invalidateQueries()
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('userEmail')
    setUser(null)

    // CRITICAL: Clear the entire React Query cache on logout.
    // Without this, the next user who logs in (or the same user after re-login)
    // sees cached data from the previous session — documents appear to be gone
    // because the old empty/stale cache is served before any refetch happens.
    queryClient.clear()
  }

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)