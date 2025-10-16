-- ============================================
-- KGP VIBES CAFÉ - SIMPLE & EASY QUERIES
-- Beginner-Friendly Essential Queries
-- ============================================

USE kgp_vibes;

-- ============================================
-- BASIC QUERIES - VIEW DATA
-- ============================================

-- 1. See all customers
SELECT * FROM customers;

-- 2. See all products
SELECT * FROM products;

-- 3. See all orders
SELECT * FROM orders;

-- 4. See all order items
SELECT * FROM order_items;

-- ============================================
-- CUSTOMER QUERIES
-- ============================================

-- 5. Find customer by email
SELECT * FROM customers 
WHERE email = 'rahul@gmail.com';

-- 6. Find customer by name
SELECT * FROM customers 
WHERE name LIKE '%Emily%';

-- 7. Count total customers
SELECT COUNT(*) as total_customers 
FROM customers;

-- ============================================
-- PRODUCT QUERIES
-- ============================================

-- 8. Show all coffee items
SELECT name, price, description 
FROM products 
WHERE category = 'COFFEE';

-- 9. Show all meals
SELECT name, price, description 
FROM products 
WHERE category = 'MEALS';

-- 10. Find products under $5
SELECT name, price, category 
FROM products 
WHERE price < 5.00
ORDER BY price;

-- 11. Search product by name
SELECT name, price, category, description 
FROM products 
WHERE name LIKE '%Latte%';

-- 12. Show most expensive items
SELECT name, price, category 
FROM products 
ORDER BY price DESC 
LIMIT 5;

-- 13. Show cheapest items
SELECT name, price, category 
FROM products 
ORDER BY price ASC 
LIMIT 5;

-- ============================================
-- ORDER QUERIES
-- ============================================

-- 14. Show all orders for a customer (by customer_id)
SELECT * FROM orders 
WHERE customer_id = 1;

-- 15. Show recent orders (last 5)
SELECT order_id, customer_id, order_date, total_amount, status 
FROM orders 
ORDER BY order_date DESC 
LIMIT 5;

-- 16. Show orders by status
SELECT * FROM orders 
WHERE status = 'Completed';

-- 17. Count orders by status
SELECT status, COUNT(*) as count 
FROM orders 
GROUP BY status;

-- ============================================
-- ORDER DETAILS (WITH NAMES)
-- ============================================

-- 18. Show order with customer name
SELECT 
    o.order_id,
    c.name as customer_name,
    o.order_date,
    o.total_amount,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_id = 1;

-- 19. Show what items are in an order
SELECT 
    p.name as product_name,
    oi.quantity,
    oi.item_price,
    (oi.quantity * oi.item_price) as subtotal
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
WHERE oi.order_id = 1;

-- 20. Show complete order details (customer + items)
SELECT 
    o.order_id,
    c.name as customer_name,
    c.email,
    p.name as product_name,
    oi.quantity,
    oi.item_price
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id = 1;

-- ============================================
-- POPULAR PRODUCTS
-- ============================================

-- 21. Most ordered products
SELECT 
    p.name,
    COUNT(*) as times_ordered
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_id
ORDER BY times_ordered DESC
LIMIT 5;

-- 22. Best selling products by quantity
SELECT 
    p.name,
    SUM(oi.quantity) as total_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_id
ORDER BY total_sold DESC
LIMIT 5;

-- ============================================
-- CUSTOMER INSIGHTS
-- ============================================

-- 23. Customer's total orders
SELECT 
    c.name,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id = 1
GROUP BY c.customer_id;

-- 24. What did a customer order before?
SELECT DISTINCT
    p.name as product_name,
    p.category,
    p.price
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.customer_id = 1;

-- 25. Customer's favorite items (most ordered)
SELECT 
    p.name,
    COUNT(*) as times_ordered
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.customer_id = 1
GROUP BY p.product_id
ORDER BY times_ordered DESC;

-- ============================================
-- SIMPLE ANALYTICS
-- ============================================

-- 26. Total revenue
SELECT SUM(total_amount) as total_revenue 
FROM orders 
WHERE status = 'Completed';

-- 27. Average order value
SELECT AVG(total_amount) as average_order 
FROM orders;

-- 28. Orders per day
SELECT 
    DATE(order_date) as date,
    COUNT(*) as orders
FROM orders
GROUP BY DATE(order_date)
ORDER BY date DESC;

-- 29. Revenue by category
SELECT 
    p.category,
    SUM(oi.quantity * oi.item_price) as revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY revenue DESC;

-- 30. Top 5 customers by spending
SELECT 
    c.name,
    c.email,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 5;

-- ============================================
-- SIMPLE RECOMMENDATIONS
-- ============================================

-- 31. Products in same category
SELECT name, price 
FROM products 
WHERE category = 'COFFEE' 
  AND product_id != 2
LIMIT 5;

-- 32. Similar priced items
SELECT name, price, category 
FROM products 
WHERE price BETWEEN 3.00 AND 4.00
  AND product_id != 2
ORDER BY price;

-- 33. Items other customers bought (simple version)
SELECT DISTINCT
    p.name,
    p.price,
    p.category
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
WHERE oi.order_id IN (
    SELECT DISTINCT order_id 
    FROM order_items 
    WHERE product_id = 2  -- Tech Latte
)
AND p.product_id != 2
LIMIT 5;

-- ============================================
-- INVENTORY CHECKS
-- ============================================

-- 34. Check stock for a product
SELECT name, stock, price 
FROM products 
WHERE name = 'Tech Latte';

-- 35. Low stock items (less than 70)
SELECT name, stock, category 
FROM products 
WHERE stock < 70
ORDER BY stock ASC;

-- 36. Out of stock items
SELECT name, category, price 
FROM products 
WHERE stock = 0 OR is_available = FALSE;

-- ============================================
-- SEARCH QUERIES
-- ============================================

-- 37. Search by keyword in name
SELECT name, price, category 
FROM products 
WHERE name LIKE '%coffee%';

-- 38. Search by keyword in description
SELECT name, price, description 
FROM products 
WHERE description LIKE '%chocolate%';

-- 39. Get all items in a price range
SELECT name, price, category 
FROM products 
WHERE price BETWEEN 3.00 AND 5.00
ORDER BY price;

-- 40. Filter by category and price
SELECT name, price, description 
FROM products 
WHERE category = 'COFFEE' 
  AND price < 4.00
ORDER BY price;

-- ============================================
-- QUICK STATS
-- ============================================

-- 41. Count products by category
SELECT 
    category, 
    COUNT(*) as item_count
FROM products
GROUP BY category;

-- 42. Total orders and revenue
SELECT 
    COUNT(*) as total_orders,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value
FROM orders;

-- 43. Customers by country
SELECT 
    country_code,
    COUNT(*) as customer_count
FROM customers
GROUP BY country_code
ORDER BY customer_count DESC;

-- ============================================
-- USEFUL COMBINATIONS
-- ============================================

-- 44. Customer with their latest order
SELECT 
    c.name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date = (
    SELECT MAX(order_date) 
    FROM orders 
    WHERE customer_id = c.customer_id
);

-- 45. Products never ordered
SELECT 
    p.name,
    p.category,
    p.price
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.product_id IS NULL;

-- ============================================
-- TESTING QUERIES (for verification)
-- ============================================

-- 46. Verify data loaded correctly
SELECT 
    (SELECT COUNT(*) FROM customers) as customers,
    (SELECT COUNT(*) FROM products) as products,
    (SELECT COUNT(*) FROM orders) as orders,
    (SELECT COUNT(*) FROM order_items) as order_items;

-- 47. Check for data issues
SELECT 'Missing customer in order' as issue, COUNT(*) as count
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

-- 48. List all tables
SHOW TABLES;

-- 49. Show table structure
DESCRIBE customers;
DESCRIBE products;
DESCRIBE orders;
DESCRIBE order_items;

-- 50. Sample of each table
SELECT 'customers' as table_name, COUNT(*) as rows FROM customers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL  
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items;

-- ============================================
-- END - 50 SIMPLE QUERIES
-- ============================================

SELECT '✅ 50 Simple queries ready to use!' as Status;
