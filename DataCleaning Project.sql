# Cleaning data in SQL queries
USE NashvilleHousing;
SELECT * FROM nh;

 
 #################################################################
 -- Populate Property Address Data
 
 SELECT *
 FROM nh
 WHERE PropertyAddress = '' OR PropertyAddress IS NULL
 ORDER BY ParcelID; #Rows with the same parcelID has the same property address, so we can fill the blank data using the same address given that
 #the entry has the same parcelID as the specific address
 
 Update nh
	SET PropertyAddress = NULL
WHERE PropertyAddress = '';
 
 
 #We use IFNULL to find NULL values in a.PropertyAddress and populate it with entries of b.PropertyAddress
 SELECT  a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, IFNULL(a.PropertyAddress, b.PropertyAddress)
 FROM nh a
 JOIN nh b
	on a.ParcelID = b.ParcelID
    AND a.uniqueID <> b.uniqueID #uniqueID represents all the different entries
WHERE b.PropertyAddress IS NULL;

#Update entries based on previous query

UPDATE nh b
	JOIN nh a
	on a.ParcelID = b.ParcelID
    AND a.uniqueID <> b.uniqueID #uniqueID represents all the different entries
SET b.PropertyAddress = IFNULL(a.PropertyAddress, b.PropertyAddress)
WHERE b.PropertyAddress IS NULL; 
 
 
 #################################################################
 
 -- Breaking out address into separate columns (Adress, City, State)
 #Using substring(string, start, end)
 SELECT PropertyAddress
 FROM nh;
 SELECT SUBSTRING(PropertyAddress,1,LOCATE(',',PropertyAddress) -1) AS Address,
	SUBSTRING(PropertyAddress,LOCATE(',',PropertyAddress) +1 ,LENGTH(PropertyAddress)) AS City
  FROM nh;
  
#Alter and update tables
ALTER TABLE nh
  ADD PropertySplitAddress Nvarchar(255);
  
UPDATE nh
  SET PropertySplitAddress = SUBSTRING(PropertyAddress,1,LOCATE(',',PropertyAddress) -1);
  
ALTER TABLE nh
  ADD PropertySplitCity Nvarchar(255);
  
UPDATE nh
  SET PropertySplitCity = SUBSTRING(PropertyAddress,LOCATE(',',PropertyAddress) +1 ,LENGTH(PropertyAddress));
 
 #See updated table
 SELECT * FROM nh;
 
 #OR use substring_index( string, 'delimiter', position) 
# this time we use the OwnerAddress column
 SELECT OwnerAddress,
 SUBSTRING_INDEX( OwnerAddress , ',' , 1) AS OwnerAddress,
 SUBSTRING_INDEX( SUBSTRING_INDEX( OwnerAddress , ',' , -2),',',  1) AS OwnerCity,
 SUBSTRING_INDEX( OwnerAddress , ',' , -1) AS OwnerState
 FROM nh
;

#Again, alter and update the table
ALTER TABLE nh
  ADD OwnerSplitAddress Nvarchar(255);
  
UPDATE nh
  SET OwnerSplitAddress =  SUBSTRING_INDEX( OwnerAddress , ',' , 1);
  
ALTER TABLE nh
  ADD OwnerSplitCity Nvarchar(255);
  
UPDATE nh
  SET OwnerSplitCity = SUBSTRING_INDEX( SUBSTRING_INDEX( OwnerAddress , ',' , -2),',',  1);
  
ALTER TABLE nh
  ADD OwnerSplitState Nvarchar(255);
  
UPDATE nh
  SET OwnerSplitState = SUBSTRING_INDEX( OwnerAddress , ',' , -1);
 
 #See updated table
 SELECT * FROM nh;
 
 #################################################################
 
 -- Change Y and N to Yes and No in "Sold as Vacant" field
 
 #Frist we find the different inputs in "sold as vacant" column
  #results show that there are 4 different answers ('N' , 'Y', 'No', and 'Yes')
 SELECT DISTINCT(SoldAsVacant), count(SoldAsVacant) FROM nh
 GROUP BY SoldAsVacant 
 ORDER BY 2;
 
 SELECT soldasvacant
 , CASE WHEN soldasvacant = 'Y' THEN 'YES'
		WHEN soldasvacant = 'N' THEN 'NO'
        ELSE soldasvacant
        END
FROM nh;

UPDATE nh
SET soldasvacant =  CASE WHEN soldasvacant = 'Y' THEN 'YES'
		WHEN soldasvacant = 'N' THEN 'NO'
        ELSE soldasvacant
        END;


SELECT DISTINCT(SoldAsVacant), count(SoldAsVacant) FROM nh
 GROUP BY SoldAsVacant 
 ORDER BY 2;
 
 #################################################################
 
 -- Remove duplicates
 #USE CTE to see duplicates
 WITH rownumCTE 
 AS 
 (
 
 SELECT *, 
 ROW_NUMBER() OVER 
	(
	PARTITION BY parcelid, 
				propertyaddress, 
				saleprice, 
                saledate, 
                legalreference
	ORDER BY uniqueid
    ) 
    AS row_num
 FROM nh
 )

SELECT uniqueid
 FROM rownumCTE
 WHERE row_num >1; #each unique entry only has 1 row_num, it has 2 row_num only if there are two of the exact same entries
 
 #Now delete the duplicates


 DELETE FROM nh
	WHERE uniqueid IN 
			( SELECT uniqueid FROM 
				(
					SELECT uniqueid, 
							ROW_NUMBER() OVER 
							(
								PARTITION BY parcelid, 
											propertyaddress, 
											saleprice, 
											saledate, 
											legalreference
								ORDER BY uniqueid
							) 
						AS row_num
					    FROM nh
				) a 
			WHERE row_num>1);
 
 #################################################################
 
 -- Delete Unused columns
 
 SELECT *
 FROM nh;
 
 ALTER TABLE nh
 DROP COLUMN OwnerAddress;
  ALTER TABLE nh
 DROP COLUMN taxdistrict;
  ALTER TABLE nh
 DROP COLUMN PropertyAddress;
  ALTER TABLE nh
 DROP COLUMN saledate;