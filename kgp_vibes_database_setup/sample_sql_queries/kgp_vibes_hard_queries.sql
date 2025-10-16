-- ============================================
-- KGP VIBES CAFÉ - SAMPLE SQL QUERIES
-- AI Assistant Query Examples
-- ============================================

USE kgp_vibes;

-- ============================================
-- 1. ORDER STATUS QUERIES
-- ============================================

-- Q1.1: Check order status for a specific customer
-- Use Case: Customer asks "Where's my order?"
SELECT 
    o.order_id,
    o.order_date,
    o.status,
    o.total_amount,
    c.name as customer_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE c.email = 'rahul@gmail.com'
ORDER BY o.order_date DESC;

-- Q1.2: Get detailed order items with product names
-- Use Case: "What did I order?"
SELECT 
    o.order_id,
    o.order_date,
    o.status,
    p.name as product_name,
    oi.quantity,
    oi.item_price,
    (oi.quantity * oi.item_price) as subtotal
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id = 1;

-- Q1.3: Get customer's order history
-- Use Case: "Show me my past orders"
SELECT 
    o.order_id,
    o.order_date,
    o.total_amount,
    o.status,
    GROUP_CONCAT(p.name SEPARATOR ', ') as items_ordered
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.customer_id = 1
GROUP BY o.order_id
ORDER BY o.order_date DESC;

-- Q1.4: Track pending/preparing orders
-- Use Case: "Do I have any pending orders?"
SELECT 
    o.order_id,
    c.name as customer_name,
    o.order_date,
    o.status,
    o.total_amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.customer_id = 2 
  AND o.status IN ('Pending', 'Preparing', 'Ready')
ORDER BY o.order_date DESC;

-- ============================================
-- 2. PRODUCT SEARCH & DISCOVERY QUERIES
-- ============================================

-- Q2.1: Search products by name
-- Use Case: "Do you have espresso?"
SELECT 
    product_id,
    name,
    category,
    description,
    price,
    stock,
    is_available
FROM products
WHERE name LIKE '%espresso%'
  AND is_available = TRUE;

-- Q2.2: Get products by category
-- Use Case: "Show me all coffee options"
SELECT 
    product_id,
    name,
    description,
    price,
    stock
FROM products
WHERE category = 'COFFEE'
  AND is_available = TRUE
ORDER BY price ASC;

-- Q2.3: Filter products by price range
-- Use Case: "What coffee can I get under $4?"
SELECT 
    product_id,
    name,
    category,
    price,
    description
FROM products
WHERE category = 'COFFEE'
  AND price < 4.00
  AND is_available = TRUE
ORDER BY price ASC;

-- Q2.4: Get all available categories
-- Use Case: "What types of items do you have?"
SELECT DISTINCT 
    category,
    COUNT(*) as item_count,
    MIN(price) as min_price,
    MAX(price) as max_price
FROM products
WHERE is_available = TRUE
GROUP BY category
ORDER BY category;

-- Q2.5: Search products across all fields
-- Use Case: "Show me anything with chocolate"
SELECT 
    product_id,
    name,
    category,
    price,
    description
FROM products
WHERE (name LIKE '%chocolate%' OR description LIKE '%chocolate%')
  AND is_available = TRUE;

-- ============================================
-- 3. PRODUCT RECOMMENDATION QUERIES
-- ============================================

-- Q3.1: Most popular products overall
-- Use Case: "What are your bestsellers?"
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    SUM(oi.quantity) as times_ordered,
    COUNT(DISTINCT oi.order_id) as order_count
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id
ORDER BY times_ordered DESC
LIMIT 5;

-- Q3.2: Products frequently bought together
-- Use Case: "People who bought X also bought Y" (Collaborative Filtering)
SELECT 
    p1.name as product_1,
    p2.name as product_2,
    COUNT(*) as times_together,
    AVG(p1.price + p2.price) as combo_price
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
    AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
GROUP BY p1.product_id, p2.product_id
HAVING times_together >= 2
ORDER BY times_together DESC
LIMIT 10;

-- Q3.3: Recommend based on what customer previously ordered
-- Use Case: "Based on your past orders, you might like..."
SELECT DISTINCT
    p.product_id,
    p.name,
    p.category,
    p.price,
    p.description
FROM products p
WHERE p.category IN (
    SELECT DISTINCT p2.category
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p2 ON oi.product_id = p2.product_id
    WHERE o.customer_id = 1
)
AND p.product_id NOT IN (
    SELECT DISTINCT oi2.product_id
    FROM orders o2
    JOIN order_items oi2 ON o2.order_id = oi2.order_id
    WHERE o2.customer_id = 1
)
AND p.is_available = TRUE
ORDER BY RAND()
LIMIT 5;

-- Q3.4: Popular items in same category
-- Use Case: "If you like Tech Latte, try these other coffees"
SELECT 
    p.product_id,
    p.name,
    p.price,
    p.description,
    COUNT(oi.order_item_id) as popularity
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.category = (
    SELECT category FROM products WHERE product_id = 2
)
AND p.product_id != 2
AND p.is_available = TRUE
GROUP BY p.product_id
ORDER BY popularity DESC, p.price ASC
LIMIT 5;

-- Q3.5: Trending products (recent orders)
-- Use Case: "What's popular this week?"
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    COUNT(oi.order_item_id) as recent_orders
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY p.product_id
ORDER BY recent_orders DESC
LIMIT 10;

-- ============================================
-- 4. CUSTOMER PROFILE & PREFERENCES
-- ============================================

-- Q4.1: Get customer information
-- Use Case: "Who is this customer?"
SELECT 
    customer_id,
    name,
    email,
    CONCAT(country_code, ' ', phone) as full_phone,
    created_at
FROM customers
WHERE email = 'rahul@gmail.com';

-- Q4.2: Customer's favorite products
-- Use Case: "What does this customer usually order?"
SELECT 
    p.product_id,
    p.name,
    p.category,
    COUNT(oi.order_item_id) as times_ordered,
    SUM(oi.quantity) as total_quantity
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.customer_id = 1
GROUP BY p.product_id
ORDER BY times_ordered DESC, total_quantity DESC
LIMIT 5;

-- Q4.3: Customer spending analysis
-- Use Case: "How much has this customer spent?"
SELECT 
    c.customer_id,
    c.name,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id = 1
GROUP BY c.customer_id;

-- Q4.4: Customer's preferred categories
-- Use Case: "What type of items does this customer prefer?"
SELECT 
    p.category,
    COUNT(oi.order_item_id) as items_ordered,
    SUM(oi.quantity) as total_quantity,
    SUM(oi.quantity * oi.item_price) as category_spending
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.customer_id = 1
GROUP BY p.category
ORDER BY items_ordered DESC;

-- ============================================
-- 5. INVENTORY & AVAILABILITY QUERIES
-- ============================================

-- Q5.1: Check if product is available
-- Use Case: "Is Database Pasta available?"
SELECT 
    product_id,
    name,
    price,
    stock,
    is_available,
    CASE 
        WHEN stock > 50 THEN 'In Stock'
        WHEN stock > 0 THEN 'Limited Stock'
        ELSE 'Out of Stock'
    END as availability_status
FROM products
WHERE name = 'Database Pasta';

-- Q5.2: Low stock alert
-- Use Case: Admin query for inventory management
SELECT 
    product_id,
    name,
    category,
    stock,
    price
FROM products
WHERE stock < 60
  AND is_available = TRUE
ORDER BY stock ASC;

-- Q5.3: Products out of stock
-- Use Case: "What items are currently unavailable?"
SELECT 
    product_id,
    name,
    category,
    price
FROM products
WHERE is_available = FALSE OR stock = 0
ORDER BY category, name;

-- ============================================
-- 6. CART & ORDER CREATION QUERIES
-- ============================================

-- Q6.1: Calculate cart total
-- Use Case: "What's my total before checkout?"
-- This would be done in application logic, but here's the SQL:
SELECT 
    SUM(p.price * cart.quantity) as cart_total
FROM (
    SELECT 2 as product_id, 1 as quantity
    UNION ALL
    SELECT 13 as product_id, 2 as quantity
    UNION ALL
    SELECT 23 as product_id, 1 as quantity
) as cart
JOIN products p ON cart.product_id = p.product_id;

-- Q6.2: Insert new order (example structure)
-- Use Case: Customer completes purchase
/*
START TRANSACTION;

-- Insert order
INSERT INTO orders (customer_id, total_amount, status) 
VALUES (1, 12.30, 'Pending');

SET @order_id = LAST_INSERT_ID();

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(@order_id, 2, 1, 3.50),
(@order_id, 13, 2, 2.50);

-- Update stock
UPDATE products SET stock = stock - 1 WHERE product_id = 2;
UPDATE products SET stock = stock - 2 WHERE product_id = 13;

COMMIT;
*/

-- Q6.3: Verify product availability before order
-- Use Case: Check if items can be fulfilled
SELECT 
    p.product_id,
    p.name,
    p.stock,
    cart.quantity,
    CASE 
        WHEN p.stock >= cart.quantity THEN 'Available'
        ELSE 'Insufficient Stock'
    END as status
FROM (
    SELECT 2 as product_id, 1 as quantity
    UNION ALL
    SELECT 13 as product_id, 2 as quantity
) as cart
JOIN products p ON cart.product_id = p.product_id;

-- ============================================
-- 7. ANALYTICS & REPORTING QUERIES
-- ============================================

-- Q7.1: Daily sales summary
-- Use Case: "How much did we make today?"
SELECT 
    DATE(order_date) as sale_date,
    COUNT(order_id) as total_orders,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE DATE(order_date) = CURDATE()
GROUP BY DATE(order_date);

-- Q7.2: Revenue by category
-- Use Case: "Which category generates most revenue?"
SELECT 
    p.category,
    COUNT(DISTINCT o.order_id) as orders,
    SUM(oi.quantity) as items_sold,
    SUM(oi.quantity * oi.item_price) as revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
GROUP BY p.category
ORDER BY revenue DESC;

-- Q7.3: Top customers by spending
-- Use Case: "Who are our best customers?"
SELECT 
    c.customer_id,
    c.name,
    c.email,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 10;

-- Q7.4: Order status distribution
-- Use Case: "How many orders are in each status?"
SELECT 
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as total_value
FROM orders
GROUP BY status
ORDER BY 
    FIELD(status, 'Pending', 'Preparing', 'Ready', 'Completed', 'Cancelled');

-- Q7.5: Average order composition
-- Use Case: "How many items do customers typically order?"
SELECT 
    AVG(item_count) as avg_items_per_order,
    MAX(item_count) as max_items,
    MIN(item_count) as min_items
FROM (
    SELECT order_id, COUNT(*) as item_count
    FROM order_items
    GROUP BY order_id
) as order_counts;

-- ============================================
-- 8. SEARCH & FILTER COMBINATIONS
-- ============================================

-- Q8.1: Complex product search
-- Use Case: "Show me coffee under $4 that's currently available"
SELECT 
    product_id,
    name,
    price,
    description,
    stock
FROM products
WHERE category = 'COFFEE'
  AND price < 4.00
  AND is_available = TRUE
  AND stock > 0
ORDER BY price ASC;

-- Q8.2: Customer's unfulfilled orders
-- Use Case: "Show orders that need attention"
SELECT 
    o.order_id,
    c.name as customer_name,
    c.email,
    c.country_code,
    c.phone,
    o.order_date,
    o.status,
    o.total_amount,
    TIMESTAMPDIFF(MINUTE, o.order_date, NOW()) as minutes_since_order
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status IN ('Pending', 'Preparing')
ORDER BY o.order_date ASC;

-- Q8.3: Product performance comparison
-- Use Case: "Compare sales of similar items"
SELECT 
    p.name,
    p.price,
    COUNT(oi.order_item_id) as times_ordered,
    SUM(oi.quantity) as total_quantity,
    SUM(oi.quantity * oi.item_price) as total_revenue,
    ROUND(SUM(oi.quantity * oi.item_price) / SUM(oi.quantity), 2) as avg_price_paid
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.category = 'COFFEE'
GROUP BY p.product_id
ORDER BY total_revenue DESC;

-- ============================================
-- 9. AI ASSISTANT SPECIFIC QUERIES
-- ============================================

-- Q9.1: Natural language to SQL - "What's cheapest on the menu?"
SELECT 
    name,
    category,
    price,
    description
FROM products
WHERE is_available = TRUE
ORDER BY price ASC
LIMIT 5;

-- Q9.2: "What can I get for under $10?"
SELECT 
    p1.name as item1,
    p1.price as price1,
    p2.name as item2,
    p2.price as price2,
    (p1.price + p2.price) as combo_price
FROM products p1
CROSS JOIN products p2
WHERE p1.product_id < p2.product_id
  AND (p1.price + p2.price) <= 10.00
  AND p1.is_available = TRUE
  AND p2.is_available = TRUE
ORDER BY combo_price DESC
LIMIT 10;

-- Q9.3: "Show me everything with 'coffee' in it"
SELECT 
    product_id,
    name,
    category,
    price,
    description
FROM products
WHERE LOWER(name) LIKE '%coffee%'
   OR LOWER(description) LIKE '%coffee%'
   OR LOWER(category) LIKE '%coffee%'
ORDER BY price ASC;

-- Q9.4: Customer lifetime value prediction
-- Use Case: Identify high-value customers for personalization
SELECT 
    c.customer_id,
    c.name,
    c.email,
    COUNT(DISTINCT o.order_id) as order_count,
    SUM(o.total_amount) as lifetime_value,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order,
    DATEDIFF(NOW(), MAX(o.order_date)) as days_since_last_order,
    CASE 
        WHEN COUNT(o.order_id) >= 5 THEN 'VIP'
        WHEN COUNT(o.order_id) >= 3 THEN 'Regular'
        ELSE 'New'
    END as customer_tier
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY lifetime_value DESC;

-- ============================================
-- 10. UTILITY QUERIES
-- ============================================

-- Q10.1: Database statistics
SELECT 
    'Customers' as table_name, COUNT(*) as record_count FROM customers
UNION ALL
SELECT 'Products', COUNT(*) FROM products
UNION ALL
SELECT 'Orders', COUNT(*) FROM orders
UNION ALL
SELECT 'Order Items', COUNT(*) FROM order_items;

-- Q10.2: Data integrity check
-- Verify all foreign key relationships are valid
SELECT 'Orders without customers' as issue, COUNT(*) as count
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL
UNION ALL
SELECT 'Order items without orders', COUNT(*)
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL
UNION ALL
SELECT 'Order items without products', COUNT(*)
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE p.product_id IS NULL;

-- ============================================
-- END OF SAMPLE QUERIES
-- ============================================

SELECT '✅ Sample queries loaded successfully!' as Status;
