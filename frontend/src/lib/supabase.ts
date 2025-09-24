import { createClient } from '@supabase/supabase-js'
import type { User } from '../types'

// Get environment variables with better error handling
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Check if environment variables are properly set
if (!supabaseUrl || supabaseUrl === 'your-supabase-url') {
  console.error('VITE_SUPABASE_URL is not set. Please create a .env.local file with your Supabase URL.')
  console.error('Example: VITE_SUPABASE_URL=https://your-project-ref.supabase.co')
}

if (!supabaseAnonKey || supabaseAnonKey === 'your-supabase-anon-key') {
  console.error('VITE_SUPABASE_ANON_KEY is not set. Please create a .env.local file with your Supabase anon key.')
  console.error('Example: VITE_SUPABASE_ANON_KEY=your-anon-key-here')
}

// Use fallback values for development
const finalSupabaseUrl = supabaseUrl || 'https://placeholder.supabase.co'
const finalSupabaseAnonKey = supabaseAnonKey || 'placeholder-key'

// Log the API URL for debugging
console.log('API URL:', import.meta.env.VITE_API_URL || 'http://localhost:5000')

export const supabase = createClient(finalSupabaseUrl, finalSupabaseAnonKey)

// Re-export types for convenience
export type { User }
