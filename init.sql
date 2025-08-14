-- Initialize the nurse_conversations database
-- This script runs automatically when PostgreSQL starts for the first time

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For full-text search if needed

-- Create indexes for better performance (will be created by SQLAlchemy migrations)
-- These are just examples - actual indexes should be in Alembic migrations

-- Sample data for testing (optional)
-- INSERT INTO conversation_uploads (id, original_filename, file_path, file_size, content_type, status)
-- VALUES (
--     uuid_generate_v4(),
--     'sample_conversation.wav',
--     '/app/uploads/sample_conversation.wav',
--     1024000,
--     'audio/wav',
--     'completed'
-- );

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Log initialization completion
SELECT 'Database initialization completed' as message;