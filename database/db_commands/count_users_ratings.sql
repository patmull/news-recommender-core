SELECT users.id, COUNT(users.id) FROM users
JOIN ratings ON users.id = ratings.user_id
GROUP BY users.id;