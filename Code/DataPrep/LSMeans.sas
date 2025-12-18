/*
Step 1: I used JMP to summarize RANGE and PASS experimental factors via Tabulate.
Step 2: I used JMP to create columns with the number of missing values for these two factors. 
Step 3: I fitted mixed models for two groups, one with environments without missing values for these
two factors, and another for all other environments. This is needed to avoid errors due to missing factors 
in proc mixed. 
Step 4: I obtained LSMeans estimates for each hybrid on all environmets and exported to data table, which will be
the response variables in the predictive models. 
*/

%let root_path = %quote(\\tdl\Public2\G2F\Manuscript\);  /*# Adjust to your path*/
%let prep_path = %quote(\Results\DataPrep\);
%let infile_path = %sysfunc(catx(,%nrstr(%superq(root_path)),%nrstr(%superq(prep_path))));
%put &infile_path;

libname g2f %quote("&infile_path");

ods html close;

/*
data full;
	set g2f._1_Training_Trait_2014_2023_S;
run;
*/

PROC IMPORT DATAFILE= %quote("%sysfunc(catx(,%nrstr(%superq(infile_path)),%nrstr(_1_Training_Trait_2014_2023_S.csv)))")
            OUT=work.full
            DBMS=CSV
            REPLACE; /* Optional: Overwrites existing dataset */
  GETNAMES=YES; /* YES if first row has headers, NO if not */
  GUESSINGROWS=MAX; /* Scans all rows for better type guessing */
RUN;

proc sort data=full;
	by env;
run;

data DEH1_2016;
	set full;
	where env = "DEH1_2016";
run;
 
ods output lsmeans=dehl_2016lsm;
proc hpmixed data=deh1_2016;
	class env year location replicate block plot hybrid range pass;
	model yield_mg_ha = hybrid replicate;
	random block(replicate) range pass;
	lsmeans hybrid;
run;

data ARH1_2017;
	set full;
	where env = "ARH1_2017";
run;


ods output lsmeans=arh1_2017lsm;
proc hpmixed data=ARH1_2017;
	class env year location replicate block hybrid /*range pass*/;
	model yield_mg_ha = hybrid replicate;
	random block(replicate) /*range pass*/;
	lsmeans hybrid;
run;

/* 
ods output lsmeans=full_lsm;
proc hpmixed data=full;
	class env year location replicate block hybrid; //range pass;
	model yield_mg_ha = hybrid replicate;
	random block(replicate) range pass;
	by env;
	lsmeans hybrid;
run;
*/
 
data allFactors misFactors;
	set full;
    if (N_Missing_Range_ > 0 | N_Missing_Pass_ > 0) then do;
		output misFactors;
	 end;
	 else do;
		output allFactors;
	 end;
run;

ods output lsmeans=AllFac_lsm;
proc hpmixed data=AllFactors;
	class env year replicate block hybrid range pass;
	model yield_mg_ha = hybrid replicate;
	random block(replicate) range pass;
	by env;
	lsmeans hybrid;
run;

ods output lsmeans=misFac_lsm;
proc hpmixed data=misFactors;
	class env year field_location replicate block hybrid /*range pass*/;
	model yield_mg_ha = hybrid replicate;
	random block(replicate)/*range pass*/;
	by env;
	lsmeans hybrid;
run;

data lsmeans;
	set allFac_lsm	misFac_lsm;
run;

/*
data g2f._1_Training_lsmeans;
	set lsmeans;
run;
*/

proc export data=lsmeans
            outfile=%quote("%sysfunc(catx(,%nrstr(%superq(infile_path)),%nrstr(_1_Training_lsmeans.csv)))")
            dbms=csv
            replace;
run;


