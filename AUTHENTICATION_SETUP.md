# Vortex AI - Authentication Setup Guide

This project now includes a complete authentication system using Supabase with React Context for state management.

## Features

- **User Registration & Login**: Complete signup and signin functionality
- **State Persistence**: Authentication state managed with React Context
- **User Profile Management**: Store user data including name, email, phone number, and reports
- **Secure Authentication**: Row Level Security (RLS) policies for data protection
- **Responsive Design**: Mobile-friendly authentication UI

## Database Schema

The Supabase schema includes:

### Users Table
- `id` (UUID, Primary Key)
- `name` (VARCHAR, Required)
- `email` (VARCHAR, Unique, Required)
- `password` (VARCHAR, Required)
- `phone_number` (VARCHAR, Optional)
- `reports` (JSONB Array, Default: [])
- `created_at` (Timestamp)
- `updated_at` (Timestamp)

### Reports Table (Optional)
- `id` (UUID, Primary Key)
- `user_id` (UUID, Foreign Key)
- `title` (VARCHAR)
- `description` (TEXT)
- `image_url` (TEXT)
- `processing_type` (VARCHAR)
- `created_at` (Timestamp)
- `updated_at` (Timestamp)

## Setup Instructions

### 1. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to the SQL Editor in your Supabase dashboard
3. Run the SQL schema from `supabase-schema.sql`
4. Go to Settings > API to get your project URL and anon key

### 2. Environment Configuration

1. Copy `env.example` to `.env.local` in the frontend directory:
   ```bash
   cp env.example .env.local
   ```

2. Update the environment variables with your Supabase credentials:
   ```env
   VITE_SUPABASE_URL=https://your-project-id.supabase.co
   VITE_SUPABASE_ANON_KEY=your-anon-key-here
   ```

### 3. Install Dependencies

The Supabase client is already installed. If you need to reinstall:

```bash
cd frontend
npm install @supabase/supabase-js
```

### 4. Run the Application

```bash
# Start the frontend
cd frontend
npm run dev

# Start the backend (in another terminal)
cd backend
python3 app.py
```

## Authentication Flow

1. **Unauthenticated Users**: See login/signup page
2. **Sign Up**: Creates user account and profile in Supabase
3. **Sign In**: Authenticates user and loads profile data
4. **Authenticated Users**: Access to the main application with user info in header
5. **Sign Out**: Clears authentication state and returns to login page

## File Structure

```
frontend/src/
├── contexts/
│   └── AuthContext.tsx          # Authentication state management
├── components/
│   ├── LoginPage.tsx            # Login form component
│   ├── SignupPage.tsx          # Signup form component
│   ├── Auth.css                # Authentication styles
│   └── ImageUpload.tsx         # Main app component
├── lib/
│   └── supabase.ts             # Supabase client configuration
└── App.tsx                     # Main app with auth integration
```

## Security Features

- **Row Level Security (RLS)**: Users can only access their own data
- **Password Hashing**: Handled by Supabase Auth
- **Session Management**: Automatic session handling with refresh tokens
- **CORS Protection**: Configured for secure cross-origin requests

## Usage

The authentication system is fully integrated into the app:

- Users must sign up or log in to access the image processing features
- User information is displayed in the header
- Authentication state persists across browser sessions
- Logout functionality clears the session and returns to login page

## Customization

You can customize the authentication system by:

1. **Styling**: Modify `Auth.css` for different visual appearance
2. **Fields**: Add more user fields in the signup form and database schema
3. **Validation**: Enhance form validation in the components
4. **Features**: Add password reset, email verification, etc.

## Troubleshooting

- **CORS Issues**: Ensure Supabase project URL is correctly configured
- **Authentication Errors**: Check Supabase project settings and RLS policies
- **Environment Variables**: Verify `.env.local` file is properly configured
- **Database Errors**: Ensure schema is correctly applied in Supabase
