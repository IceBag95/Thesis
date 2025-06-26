import * as React from 'react';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import FormContainer from './form_container';


export default function BaseContainer() {
  const [steps, setSteps] = React.useState([]);
  const [result, setResult] = React.useState({});
  const [activeStep, setActiveStep] = React.useState(0);
  const [userAnswers, setUserAnwers] = React.useState({ "usr_ans_list": []});
  const [currentAns, setCurrentAns] = React.useState({});
  const [hasResults, setHasResults] = React.useState(false);
  const [canGoNext, setCanGoNext] = React.useState(false);


  const handleNext = () => {
    let currentStep = activeStep;

    let myAnswer = userAnswers.usr_ans_list.find((ans) => ans.for_column == currentAns.for_column);
    let ansPos = userAnswers.usr_ans_list.indexOf(myAnswer)
    
    // Handle total userAnswers array
    let updatedUserAnswers = userAnswers;
    if (ansPos == -1) {
      updatedUserAnswers.usr_ans_list.push(currentAns);
      setUserAnwers(updatedUserAnswers);
    }
    else {
      updatedUserAnswers.usr_ans_list[ansPos].current_answer = currentAns.current_answer;
      setUserAnwers(updatedUserAnswers);
    }
    
    // Handle current ans that needs to be loaded for the user, either empty if 
    // next activeStep is not in the Array yet, or the respective answer if it is.
    if (currentStep + 1 >= userAnswers.usr_ans_list.length) {
      setCurrentAns({});
    }
    else {
      setCurrentAns(userAnswers.usr_ans_list[currentStep + 1]);
    }

    setCanGoNext(false);
    setActiveStep((prevActiveStep) => prevActiveStep + 1);

  };

  const handleBack = () => {
    let prevActiveStep = activeStep - 1;
    setCurrentAns(userAnswers.usr_ans_list[prevActiveStep]);
    setActiveStep(prevActiveStep);
  };

  const handleReset = () => {
    setActiveStep(0);
    setUserAnwers({ "usr_ans_list": []});
    setCurrentAns({});
    setResult({});
    setHasResults(false);
  };

  React.useEffect(() => {
      const fetchData = async () => {
        if (activeStep === steps.length && steps.length > 0) {
          try {
            const response = await fetch("http://localhost:8000/makeprediction", {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(userAnswers),
            });
            if (response.ok) {
              const result = await response.json();
              console.log("Fetched Data:", result);
              setResult(result);
            } else {
              console.error("Failed to fetch data:", response.statusText);
            }
          } catch (error) {
            console.error("Error fetching data:", error);
          }

          setHasResults(true);

        }
      }
    
      fetchData();
  }, [activeStep])

  return (
    <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
      <Stepper activeStep={activeStep} sx={{width: '50%'}}>
        {steps.length > 0 ? steps.map((label) => {
          const stepProps = {};
          const labelProps = {};
          return (
            <Step key={label} {...stepProps}>
              <StepLabel {...labelProps}></StepLabel>
            </Step>
          )
        }) : null}
      </Stepper>
      {steps.length > 0 && activeStep === steps.length ? (
        <React.Fragment>
          { hasResults ? 
            <Typography sx={{ mt: 2, mb: 1 }}>
              {
                <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
                  <p>Η διαδικασία ολοκληρώθηκε με αποτέλεσμα:</p>
                  {
                    result.target ? 
                      <h1 style={{color: 'red'}}> Υψηλή Πιθανότητα</h1>
                    : 
                      <h1 style={{color: 'green'}}> Χαμηλή Πιθανότητα</h1>
                  }
                  {
                    result.target ? 
                      <h3>Σύμφωνα με τα δεδομένα που εισήχθησαν ανιχνεύθηκε υψηλή πιθανότητα καρδιακής προσβολής</h3>
                    : 
                      <h3>Σύμφωνα με τα δεδομένα που εισήχθησαν προκύπτει χαμηλή πιθανότητα καρδιακής προσβολής</h3>
                  }
                </div>
              }
            </Typography>
           : 
           <Typography sx={{ mt: 2, mb: 1 }}> 
            {
              <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
                <p>Περιμένετε να υπλογιστούν τα αποτελέσματα...</p>
              </div> 
              }
          </Typography>
          }
          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button onClick={handleReset}>Νεα Προβλεψη</Button>
          </Box>
        </React.Fragment>
      ) : (
        <React.Fragment sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          <Typography sx={{ mt: 2, mb: 1 , display: 'flex', width: '60%'}}>
            <FormContainer step={activeStep} currAns={currentAns} setCurrAns={setCurrentAns} setNoQ={setSteps} canGoNext={canGoNext} setCanGoNext={setCanGoNext}/>          
          </Typography>
          <Box sx={{ display: 'flex', pt: 2 , width: '50%', justifyContent:'space-evenly' , alignItems: 'center'}}>
            <Button
              color="inherit"
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ margin: "0px 20px" }}
            >
              Πισω
            </Button>
            <Box sx={{ display: 'flex' }} />
                <Button onClick={handleNext} sx={{ margin: "0px 20px" }} disabled={!canGoNext}>
                {activeStep === steps.length - 1 ? 'Αποστολη απαντησεων' : 'Επομενο'}
                </Button>
          </Box>
        </React.Fragment>
      )}
    </Box>
  );
}
