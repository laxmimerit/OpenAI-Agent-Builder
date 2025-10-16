-- ============================================
-- KGP VIBES CAFÉ DATABASE SETUP
-- Database: kgp_vibes
-- ============================================

-- Create Database
CREATE DATABASE IF NOT EXISTS kgp_vibes;
USE kgp_vibes;

-- ============================================
-- TABLE SCHEMAS
-- ============================================

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- Create Customers Table
CREATE TABLE customers (
  customer_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  country_code VARCHAR(10),
  phone VARCHAR(20),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Products Table
CREATE TABLE products (
  product_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  category VARCHAR(50),
  description TEXT,
  price DECIMAL(10,2),
  stock INT DEFAULT 100,
  is_available BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Orders Table
CREATE TABLE orders (
  order_id INT AUTO_INCREMENT PRIMARY KEY,
  customer_id INT,
  order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  total_amount DECIMAL(10,2),
  status ENUM('Pending', 'Preparing', 'Ready', 'Completed', 'Cancelled') DEFAULT 'Pending',
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create Order Items Table
CREATE TABLE order_items (
  order_item_id INT AUTO_INCREMENT PRIMARY KEY,
  order_id INT,
  product_id INT,
  quantity INT,
  item_price DECIMAL(10,2),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- ============================================
-- INSERT CUSTOMER DATA (10 Customers)
-- ============================================

INSERT INTO customers (name, email, country_code, phone) VALUES
('Rahul Kumar', 'rahul@gmail.com', '+91', '9876543210'),
('Emily Chen', 'emily.chen@outlook.com', '+1', '5550123456'),
('Ahmed Al-Farsi', 'ahmed.alfarsi@yahoo.com', '+971', '501234567'),
('Yuki Tanaka', 'yuki.tanaka@icloud.com', '+81', '9012345678'),
('Priya Singh', 'priya.singh@hotmail.com', '+91', '9123456789'),
('Marcus Johnson', 'marcus.j@protonmail.com', '+44', '7700900123'),
('Sofia Rodriguez', 'sofia.r@gmail.com', '+34', '612345678'),
('Chen Wei', 'chen.wei@163.com', '+86', '13812345678'),
('Isabella Rossi', 'bella.rossi@libero.it', '+39', '3201234567'),
('David Kim', 'david.kim@naver.com', '+82', '1012345678');

-- ============================================
-- INSERT PRODUCT DATA (All KGP Vibes Menu Items)
-- ============================================

-- COFFEE Category
INSERT INTO products (name, category, description, price, stock) VALUES
('Vibes Espresso Shot', 'COFFEE', 'Classic strong espresso to kickstart your coding brain.', 2.50, 100),
('Tech Latte', 'COFFEE', 'Smooth latte with vanilla and caramel.', 3.50, 100),
('KGP Cold Brew', 'COFFEE', '8-hour brewed chill coffee — bold & smooth.', 4.00, 80),
('Code Cappuccino', 'COFFEE', 'Perfectly balanced foam & espresso.', 3.20, 100),
('Debug Mocha', 'COFFEE', 'Chocolate-infused mocha for late-night fixes.', 3.80, 90),
('Hackathon Iced Coffee', 'COFFEE', 'Sweet and energetic — keeps you awake.', 4.50, 85),
('Caramel Macchiato', 'COFFEE', 'Layered espresso with vanilla and caramel drizzle.', 4.20, 95),
('Flat White', 'COFFEE', 'Velvety microfoam meets rich espresso.', 3.60, 100),
('Talkie Chai', 'COFFEE', 'Spiced Indian chai, café's soul drink.', 2.00, 120),
('Matcha Latte', 'COFFEE', 'Premium Japanese green tea with steamed milk.', 4.00, 70),
('Nitro Cold Brew', 'COFFEE', 'Smooth, creamy cold brew infused with nitrogen.', 4.50, 60),
('Campus Hot Chocolate', 'COFFEE', 'Creamy hot chocolate with marshmallows.', 3.00, 100);

-- SNACKS & MEALS Category
INSERT INTO products (name, category, description, price, stock) VALUES
('Byte Brownie', 'SNACKS & MEALS', 'Gooey chocolate brownie — perfect with coffee.', 2.50, 80),
('Python Pancakes', 'SNACKS & MEALS', 'Fluffy stack with maple syrup & butter.', 4.50, 50),
('Caffeine Sandwich', 'SNACKS & MEALS', 'Grilled veggie sandwich with house sauce.', 3.80, 70),
('KGP Maggi Bowl', 'SNACKS & MEALS', '2-minute nostalgia — cheese or spicy.', 2.20, 100),
('AI Waffle', 'SNACKS & MEALS', 'Crispy waffle with Nutella & banana.', 4.00, 60),
('Binary Bagel', 'SNACKS & MEALS', 'Toasted bagel with cream cheese & herbs.', 3.20, 75),
('Loop Croissant', 'SNACKS & MEALS', 'Buttery French croissant, flaky perfection.', 3.00, 80),
('Stack Overflow Nachos', 'SNACKS & MEALS', 'Loaded nachos with cheese & jalapeños.', 5.50, 55),
('Git Commit Cookies', 'SNACKS & MEALS', 'Freshly baked chocolate chip cookies.', 2.80, 90);

-- MEALS Category
INSERT INTO products (name, category, description, price, stock) VALUES
('Startup Burger', 'MEALS', 'Juicy patty, lettuce, cheese — startup fuel.', 5.00, 60),
('Cloud Fries', 'MEALS', 'French fries with peri-peri seasoning.', 2.80, 100),
('Talkie Tacos', 'MEALS', 'Soft tacos with paneer or chicken filling.', 4.50, 65),
('Vibes Pizza Slice', 'MEALS', 'Thin crust slice — cheesy & herby.', 4.00, 70),
('Algorithm Wrap', 'MEALS', 'Grilled wrap with hummus, veggies & falafel.', 5.20, 60),
('Database Pasta', 'MEALS', 'Creamy alfredo pasta with garlic bread.', 6.50, 50),
('Server Side Salad', 'MEALS', 'Fresh garden salad with balsamic dressing.', 4.80, 65),
('Backend Bowl', 'MEALS', 'Quinoa bowl with roasted veggies & tahini.', 5.80, 55),
('Full Stack Quesadilla', 'MEALS', 'Cheesy quesadilla with salsa & sour cream.', 5.50, 60);

-- DESSERTS & DRINKS Category
INSERT INTO products (name, category, description, price, stock) VALUES
('Chill Zone Mojito', 'DESSERTS & DRINKS', 'Mint & lime mocktail that refreshes.', 3.20, 80),
('Data Stream Smoothie', 'DESSERTS & DRINKS', 'Mango-banana smoothie with protein.', 3.80, 75),
('Zero-Bug Brownie Sundae', 'DESSERTS & DRINKS', 'Ice cream + brownie + fudge = perfect.', 4.20, 60),
('Compile Cheesecake', 'DESSERTS & DRINKS', 'Rich New York style cheesecake slice.', 4.50, 50),
('Cache Milkshake', 'DESSERTS & DRINKS', 'Thick vanilla or chocolate milkshake.', 4.00, 70),
('API Affogato', 'DESSERTS & DRINKS', 'Vanilla gelato drowned in hot espresso.', 4.80, 45),
('Bandwidth Lemonade', 'DESSERTS & DRINKS', 'Fresh-squeezed lemonade, sweet & tangy.', 2.80, 90),
('Runtime Tiramisu', 'DESSERTS & DRINKS', 'Classic Italian coffee-soaked dessert.', 5.00, 40),
('Frontend Fruit Bowl', 'DESSERTS & DRINKS', 'Seasonal fresh fruit with honey drizzle.', 4.20, 60);

-- ============================================
-- INSERT ORDERS WITH OVERLAPPING PRODUCTS
-- (Designed for recommendation engine testing)
-- ============================================

-- Order 1: Rahul Kumar (Popular coffee combo)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(1, '2025-10-14 09:30:00', 12.30, 'Completed');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(1, 2, 2, 3.50),  -- 2x Tech Latte
(1, 13, 1, 2.50), -- 1x Byte Brownie
(1, 23, 1, 2.80); -- 1x Cloud Fries

-- Order 2: Emily Chen (Coffee + Dessert lover)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(2, '2025-10-14 11:45:00', 11.50, 'Completed');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(2, 2, 1, 3.50),  -- 1x Tech Latte (OVERLAP with Order 1)
(2, 33, 1, 4.20), -- 1x Zero-Bug Brownie Sundae
(2, 35, 1, 4.00); -- 1x Cache Milkshake

-- Order 3: Ahmed Al-Farsi (Premium coffee enthusiast)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(3, '2025-10-15 08:15:00', 15.70, 'Completed');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(3, 6, 2, 4.50),  -- 2x Hackathon Iced Coffee
(3, 27, 1, 6.50), -- 1x Database Pasta
(3, 37, 1, 2.80); -- 1x Bandwidth Lemonade

-- Order 4: Yuki Tanaka (Matcha & Japanese style)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(4, '2025-10-15 10:00:00', 13.00, 'Completed');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(4, 10, 2, 4.00), -- 2x Matcha Latte
(4, 38, 1, 5.00); -- 1x Runtime Tiramisu

-- Order 5: Priya Singh (Chai lover + Indian favorites)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(5, '2025-10-15 14:30:00', 10.40, 'Completed');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(5, 9, 3, 2.00),  -- 3x Talkie Chai
(5, 16, 2, 2.20); -- 2x KGP Maggi Bowl

-- Order 6: Marcus Johnson (Hearty meal combo)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(6, '2025-10-16 12:15:00', 17.30, 'Ready');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(6, 22, 1, 5.00),  -- 1x Startup Burger
(6, 23, 2, 2.80),  -- 2x Cloud Fries (OVERLAP with Order 1)
(6, 6, 1, 4.50),   -- 1x Hackathon Iced Coffee (OVERLAP with Order 3)
(6, 13, 1, 2.50);  -- 1x Byte Brownie (OVERLAP with Order 1)

-- Order 7: Sofia Rodriguez (Coffee + Pastry)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(7, '2025-10-16 09:00:00', 9.70, 'Preparing');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(7, 2, 1, 3.50),  -- 1x Tech Latte (OVERLAP with Orders 1, 2)
(7, 19, 1, 3.00), -- 1x Loop Croissant
(7, 21, 1, 2.80); -- 1x Git Commit Cookies

-- Order 8: Chen Wei (Cold drinks enthusiast)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(8, '2025-10-16 15:30:00', 12.50, 'Preparing');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(8, 3, 1, 4.00),  -- 1x KGP Cold Brew
(8, 11, 1, 4.50), -- 1x Nitro Cold Brew
(8, 31, 1, 3.20), -- 1x Chill Zone Mojito
(8, 37, 1, 2.80); -- 1x Bandwidth Lemonade (OVERLAP with Order 3)

-- Order 9: Isabella Rossi (Italian favorites)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(9, '2025-10-16 13:00:00', 13.80, 'Pending');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(9, 25, 1, 4.00), -- 1x Vibes Pizza Slice
(9, 38, 1, 5.00), -- 1x Runtime Tiramisu (OVERLAP with Order 4)
(9, 36, 1, 4.80); -- 1x API Affogato

-- Order 10: David Kim (Budget-friendly combo)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(10, '2025-10-16 16:45:00', 9.50, 'Pending');
INSERT INTO order_items (order_id, product_id, quantity, item_price) VALUES
(10, 9, 2, 2.00),  -- 2x Talkie Chai (OVERLAP with Order 5)
(10, 16, 1, 2.20), -- 1x KGP Maggi Bowl (OVERLAP with Order 5)
(10, 21, 1, 2.80); -- 1x Git Commit Cookies (OVERLAP with Order 7)

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Check all customers
SELECT * FROM customers;

-- Check products by category
SELECT category, COUNT(*) as product_count FROM products GROUP BY category;

-- Check orders summary
SELECT 
    o.order_id,
    c.name as customer_name,
    o.order_date,
    o.total_amount,
    o.status,
    COUNT(oi.order_item_id) as items_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id;

-- Most popular products (for recommendations)
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    SUM(oi.quantity) as total_ordered,
    COUNT(DISTINCT oi.order_id) as order_count
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id
ORDER BY total_ordered DESC
LIMIT 10;

-- Products frequently bought together (recommendation analysis)
SELECT 
    p1.name as product_1,
    p2.name as product_2,
    COUNT(*) as times_ordered_together
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
GROUP BY p1.product_id, p2.product_id
ORDER BY times_ordered_together DESC
LIMIT 15;

-- ============================================
-- DATABASE SETUP COMPLETE
-- ============================================

SELECT 'Database kgp_vibes created successfully!' as Status;
SELECT COUNT(*) as total_customers FROM customers;
SELECT COUNT(*) as total_products FROM products;
SELECT COUNT(*) as total_orders FROM orders;
SELECT COUNT(*) as total_order_items FROM order_items;
