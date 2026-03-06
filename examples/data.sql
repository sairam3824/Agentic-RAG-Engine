CREATE TABLE IF NOT EXISTS kpi_metrics (
  id INTEGER PRIMARY KEY,
  quarter TEXT,
  revenue_usd_mn REAL,
  gross_margin REAL,
  region TEXT
);

INSERT INTO kpi_metrics (quarter, revenue_usd_mn, gross_margin, region) VALUES
('2025-Q1', 120.5, 0.42, 'North America'),
('2025-Q2', 131.2, 0.44, 'Europe'),
('2025-Q3', 138.9, 0.45, 'Asia-Pacific'),
('2025-Q4', 149.1, 0.47, 'North America');
