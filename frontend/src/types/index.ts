// User types
export type User = {
  id: string
  name: string
  email: string
  phone_number?: string
  reports: any[]
  created_at: string
  updated_at: string
}

// Supabase User type for auth
export interface SupabaseUser {
  id: string
  email?: string
  user_metadata?: {
    name?: string
    phone_number?: string
  }
}
