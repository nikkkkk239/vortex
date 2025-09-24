import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { supabase } from '../lib/supabase'
import type { User, SupabaseUser } from '../types'

interface AuthContextType {
  user: User | null
  authUser: SupabaseUser | null
  session: any | null
  loading: boolean
  signUp: (email: string, password: string, name: string, phoneNumber?: string) => Promise<{ error: any }>
  signIn: (email: string, password: string) => Promise<{ error: any }>
  signOut: () => Promise<void>
  updateProfile: (updates: Partial<User>) => Promise<{ error: any }>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [authUser, setAuthUser] = useState<SupabaseUser | null>(null)
  const [session, setSession] = useState<any | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      if (session?.user) {
        setAuthUser(session.user)
        fetchUserProfile(session.user.id)
      } else {
        setLoading(false)
      }
    })

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      setSession(session)
      if (session?.user) {
        setAuthUser(session.user)
        await fetchUserProfile(session.user.id)
      } else {
        setAuthUser(null)
        setUser(null)
        setLoading(false)
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  const fetchUserProfile = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .single()

      if (error) {
        console.error('Error fetching user profile:', error)
        
        // If user profile doesn't exist, create it
        if (error.code === 'PGRST116') { // No rows returned
          const { data: authUser } = await supabase.auth.getUser()
          if (authUser.user) {
            const { error: insertError } = await supabase
              .from('users')
              .insert({
                id: authUser.user.id,
                name: authUser.user.user_metadata?.name || 'User',
                email: authUser.user.email || '',
                phone_number: authUser.user.user_metadata?.phone_number,
                reports: [],
              })
            
            if (insertError) {
              console.error('Error creating user profile:', insertError)
              setLoading(false)
              return
            }
            
            // Fetch the newly created profile
            const { data: newProfile } = await supabase
              .from('users')
              .select('*')
              .eq('id', userId)
              .single()
            
            setUser(newProfile)
          }
        } else {
          setLoading(false)
          return
        }
      } else {
        setUser(data)
      }
    } catch (error) {
      console.error('Error fetching user profile:', error)
    } finally {
      setLoading(false)
    }
  }

  const signUp = async (email: string, password: string, name: string, phoneNumber?: string) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            name,
            phone_number: phoneNumber,
          },
        },
      })

      if (error) {
        return { error }
      }

      // Note: User profile will be created after email confirmation
      // The user needs to confirm their email before they can be authenticated
      // and the RLS policy will allow profile creation
      
      return { error: null }
    } catch (error) {
      return { error }
    }
  }

  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })
      return { error }
    } catch (error) {
      return { error }
    }
  }

  const signOut = async () => {
    try {
      await supabase.auth.signOut()
    } catch (error) {
      console.error('Error signing out:', error)
    }
  }

  const updateProfile = async (updates: Partial<User>) => {
    if (!user) return { error: new Error('No user logged in') }

    try {
      const { error } = await supabase
        .from('users')
        .update(updates)
        .eq('id', user.id)

      if (error) {
        return { error }
      }

      // Update local state
      setUser({ ...user, ...updates })
      return { error: null }
    } catch (error) {
      return { error }
    }
  }

  const value = {
    user,
    authUser,
    session,
    loading,
    signUp,
    signIn,
    signOut,
    updateProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
