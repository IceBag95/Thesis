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
  const [skipped, setSkipped] = React.useState(new Set());
  const [userAnswers, setUserAnwers] = React.useState({ "usr_ans_list": []});
  const [currentAns, setCurrentAns] = React.useState({});


  const isStepSkipped = (step) => {
    return skipped.has(step);
  };

  const handleNext = () => {
    let newSkipped = skipped;
    let currentStep = activeStep;
    if (isStepSkipped(activeStep)) {
      newSkipped = new Set(newSkipped.values());
      newSkipped.delete(activeStep);
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped(newSkipped);

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
    if (activeStep + 1 >= userAnswers.usr_ans_list.length) {
      setCurrentAns({});
    }
    else {
      setCurrentAns(userAnswers.usr_ans_list[currentStep + 1]);
    }

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
  };

  React.useEffect(() => {
      const fetchData = async () => {
        if (activeStep === steps.length - 1) {
          try {
            const response = await fetch("http://localhost:8000/makeprediction");
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
        }
      };
    
      fetchData();
  }, [activeStep])

  return (
    <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
      <Stepper activeStep={activeStep} sx={{width: '50%'}}>
        {steps.length > 0 ? steps.map((label, index) => {
          const stepProps = {};
          const labelProps = {};
          if (isStepSkipped(index)) {
            stepProps.completed = false;
          }
          return (
            <Step key={label} {...stepProps}>
              <StepLabel {...labelProps}></StepLabel>
            </Step>
          )
        }) : null}
      </Stepper>
      {steps.length > 0 && activeStep === steps.length ? (
        <React.Fragment>
          <Typography sx={{ mt: 2, mb: 1 }}>
            All steps completed - you&apos;re finished
            {/* Here we add the async logic with the prediction */
              <h3>{JSON.stringify(result)}</h3>
            }
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button onClick={handleReset}>Νέα Πρόβλεψη</Button>
          </Box>
        </React.Fragment>
      ) : (
        <React.Fragment sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          <Typography sx={{ mt: 2, mb: 1 , display: 'flex', width: '60%'}}>
            <FormContainer step={activeStep} currAns={currentAns} setCurrAns={setCurrentAns} setNoQ={setSteps}/>          
          </Typography>
          <Box sx={{ display: 'flex', pt: 2 , width: '50%', justifyContent:'space-evenly' , alignItems: 'center'}}>
            <Button
              color="inherit"
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ margin: "0px 20px" }}
            >
              Back
            </Button>
            <Box sx={{ display: 'flex' }} />
                <Button onClick={handleNext} sx={{ margin: "0px 20px" }} disabled={!currentAns.current_answer}>
                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                </Button>
          </Box>
        </React.Fragment>
      )}
    </Box>
  );
}
