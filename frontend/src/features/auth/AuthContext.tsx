import React, { createContext, useContext, useState, useEffect } from 'react'

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

  const login = (token: string, id: number, email: string) => {
    localStorage.setItem('token', token)
    localStorage.setItem('userId', String(id))
    localStorage.setItem('userEmail', email)
    setUser({ token, id, email })
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('userEmail')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
