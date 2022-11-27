SELECT users.id, COUNT(users.id) FROM users
JOIN user_histories ON users.id = user_histories.user_id
GROUP BY users.id;