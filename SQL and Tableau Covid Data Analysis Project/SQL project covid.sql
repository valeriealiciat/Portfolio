USE  Covid;
SELECT * FROM coviddeaths
ORDER BY 3,4;

SELECT * FROM covidvaccinations
ORDER BY 3,4;

SELECT location, date, total_cases, new_cases, total_deaths, population
	FROM coviddeaths
    order by 1,2;

#1. Looking at Total cases vs total deaths
SELECT SUM(new_cases) as total_cases, SUM(new_deaths) as total_deaths, (SUM(new_deaths)/SUM(new_cases)) *100 AS Death_Percentage
	FROM coviddeaths
    WHERE continent NOT LIKE '' # '%asia%'OR continent LIKE '%africa%'OR continent LIKE '%america%'OR continent LIKE '%antarctica%'OR continent LIKE '%europe%' OR continent LIKE '%australia%' or continent LIKE '%oceania%'
    #WHERE location like '%states%'
;
    
    #Looking at Total Cases vs Population
    #shows what percentage of population got covid
SELECT location, date, total_cases, population, (total_cases/population) *100 AS Cases_Percentage
	FROM coviddeaths
    #WHERE location like '%states%'
    order by location, date;
    
#Looking at countries with highest infection rate compared to population

SELECT location, population, MAX(total_cases) as Highest_Infection_Count, MAX( (total_cases/population)) *100 AS Highest_Infection_Percentage
	FROM coviddeaths
    #WHERE location like '%states%'
    GROUP BY population, location
    order by Highest_Infection_Percentage DESC;

#Showing countries with Highest Death Count per Population
# using like and or because IS NOT NULL does not work
SELECT location, continent, MAX(total_deaths) AS Total_Death_Count from coviddeaths
WHERE continent NOT LIKE '' # '%asia%'OR continent LIKE '%africa%'OR continent LIKE '%america%'OR continent LIKE '%antarctica%'OR continent LIKE '%europe%' OR continent LIKE '%australia%' or continent LIKE '%oceania%'
GROUP BY location, continent
ORDER BY Total_Death_Count DESC;

#Lets break things down by continent
#Showing continents with the highest death count per population
SELECT  location, MAX(total_deaths) AS Total_Death_Count from coviddeaths
WHERE continent LIKE '' 
GROUP BY location
ORDER BY Total_Death_Count DESC;

#cases and death rate accumulated per day
SELECT date, SUM(new_cases) as Total_cases, SUM(new_deaths) as Total_deaths, SUM(new_deaths)*100/SUM(new_cases) as Death_Percentage#, total_deaths, (total_deaths/total_cases) *100 AS Death_Percentage
	FROM coviddeaths
	WHERE continent NOT LIKE '' 
    GROUP by date
    ORDER by date ASC;
    
#Total cases, total death, and total death per cases percentage
SELECT SUM(new_cases) as Total_cases, SUM(new_deaths) as Total_deaths, SUM(new_deaths)*100/SUM(new_cases) as Death_Percentage#, total_deaths, (total_deaths/total_cases) *100 AS Death_Percentage
	FROM coviddeaths
	WHERE continent NOT LIKE ''    ;
  
#Looking at Total population vs Vaccinations
#use cte
WITH popsvac (continent, location, date, population, new_vaccinations, Accumulated_vaccinated)
AS
(
SELECT cd.continent, 
		cd.location, 
        cd.date, 
        cd.population, 
        cv.new_vaccinations,
        SUM(cv.new_vaccinations) 
        
OVER 
	(
    Partition by cd.location 
    ORDER BY cd.location, 
			cd.date
	) 
	AS Accumulated_vaccinated  #order by location and date so that the sum is accumulated according to the date
FROM coviddeaths cd
	JOIN covidvaccinations cv
    ON cd.location = cv.location
    AND cd.date = cv.date
WHERE cd.continent LIKE '%asia%' OR cd.continent LIKE '%africa%'OR cd.continent LIKE '%america%' OR cd.continent LIKE '%antarctica%' OR cd.continent LIKE '%europe%' OR cd.continent LIKE '%australia%' OR cd.continent LIKE '%oceania%'
#ORDER BY 2,3
)
SELECT *, 
		(Accumulated_vaccinated/population) * 100 AS vaccinated_population_percentage
FROM popsvac;

#TEMPORARY TABLE

DROP table if exists PercentPopulationVaccinated;
CREATE table PercentPopulationVaccinated (
	continent nvarchar(255),
	location nvarchar(255),
	Date datetime,
	population numeric,
	new_vaccinations int,
	Accumulated_vaccinated numeric
);

INSERT INTO PercentPopulationVaccinated 
SELECT cd.continent, cd.location, cd.date, cd.population, cv.new_vaccinations 
,SUM(cv.new_vaccinations) OVER (Partition by cd.location ORDER BY cd.location, cd.date) AS Accumulated_vaccinated #order by location and date so that the sum is accumulated according to the date
FROM coviddeaths cd
	JOIN covidvaccinations cv
    ON cd.location = cv.location
    AND cd.date = cv.date
#WHERE cd.continent LIKE '%asia%' OR cd.continent LIKE '%africa%'OR cd.continent LIKE '%america%' OR cd.continent LIKE '%antarctica%' OR cd.continent LIKE '%europe%' OR cd.continent LIKE '%australia%' OR cd.continent LIKE '%oceania%'
#ORDER BY 2,3
; 
