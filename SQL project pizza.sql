#Total revenue of pizza
SELECT SUM(total_price) AS total_revenue FROM pizza_sales;

#Average order value per order
SELECT SUM(total_price)/COUNT(DISTINCT order_id) AS avg_order_value FROM pizza_sales;

#Total pizza sold
SELECT SUM(quantity) AS total_pizzas_sold FROM pizza_sales;

#Total orders
SELECT COUNT(DISTINCT order_id) AS total_orders FROM pizza_sales;

#Average pizza per order
SELECT SUM(quantity)/COUNT(DISTINCT order_id) AS Avg_Pizzas_per_order FROM pizza_sales ; 

#Hourly Trend For Total Orders
SELECT HOUR(order_time) as order_time, 
	COUNT(DISTINCT order_id) as total_orders FROM pizza_sales 
			GROUP BY HOUR(order_time) 
            ORDER BY order_time ASC;
            
#Daily Trend For Total Orders
SELECT DAYNAME(order_date) as order_day, 
	COUNT(DISTINCT order_id) as total_orders FROM pizza_sales 
			GROUP BY DAYNAME(order_date) ;
            
#Monthly Trend for total orders
SELECT MONTHNAME(order_date) AS month_name, 
	COUNT(DISTINCT order_id) AS total_orders FROM pizza_sales
		GROUP BY month_name
		ORDER BY total_orders DESC;

#Percentage of sales by pizza category	
SELECT pizza_category, SUM(total_price) as total_sales, SUM(total_price) *100 / 
	(SELECT SUM(total_price) from pizza_sales WHERE MONTH(order_date)=1 ) AS Percentage_of_sales_by_category  from pizza_sales #total sale by category / total sales (klo ad monthnya brti total sales nya jg hrs in tht month)
	
    WHERE MONTH(order_date)=1 #percentage of sales by pizza category in January (1)
    GROUP BY pizza_category;

#percentage of Sales by pizza size
SELECT pizza_size, FORMAT(SUM(total_price),2) as total_sales, FORMAT(SUM(total_price) *100 / 
	(SELECT SUM(total_price) from pizza_sales WHERE QUARTER(order_date)=1 ), 2) AS Percentage_of_sales_by_size  
    from pizza_sales #total sale by size / total sales (klo ad monthnya brti total sales nya jg hrs in tht quarter)
	
    WHERE QUARTER(order_date)=1 #percentage of sales by pizza size in 1st quarter of the year
    GROUP BY pizza_size;

#Bottom 5 Best sellers by *Revenue*, Total Quantity, Total Orders
SELECT pizza_name, SUM(total_price) AS Total_revenue FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_revenue ASC
    LIMIT 5;
    
#Top 5 Best sellers by *Revenue*, Total Quantity, Total Orders
SELECT pizza_name, SUM(total_price) AS Total_revenue FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_revenue DESC
    LIMIT 5;
    
#Top 5 Best sellers by Revenue, *Total Quantity*, Total Orders
SELECT pizza_name, SUM(quantity) AS Total_quantity FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_quantity DESC
    LIMIT 5;

#Worst 5 Best sellers by Revenue, *Total Quantity*, Total Orders
SELECT pizza_name, SUM(quantity) AS Total_quantity FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_quantity ASC
    LIMIT 5;
    
#Top 5 Best sellers by Revenue, Total Quantity, *Total Orders*
SELECT pizza_name, COUNT(DISTINCT order_id) AS Total_orders FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_orders DESC
    LIMIT 5;
    
#Worst 5 Best sellers by Revenue, Total Quantity, *Total Orders*
SELECT pizza_name, COUNT(DISTINCT order_id) AS Total_orders FROM pizza_sales
	GROUP BY pizza_name
    ORDER BY Total_orders ASC
    LIMIT 5;