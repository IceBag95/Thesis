import * as React from 'react';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import FormContainer from './form_container';

const makeNumberOfQuestions = () => {
    let questionLabelArray = [];
    for(let i = 0; i < 12; i++) {
        questionLabelArray.push(`Question ${i+1}`)
    }
    return [...questionLabelArray];
}

const steps = makeNumberOfQuestions();

export default function BaseContainer() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [skipped, setSkipped] = React.useState(new Set());


  const isStepSkipped = (step) => {
    return skipped.has(step);
  };

  const handleNext = () => {
    let newSkipped = skipped;
    if (isStepSkipped(activeStep)) {
      newSkipped = new Set(newSkipped.values());
      newSkipped.delete(activeStep);
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped(newSkipped);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  return (
    <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'}}>
      <Stepper activeStep={activeStep} sx={{width: '50%'}}>
        {steps.map((label, index) => {
          const stepProps = {};
          const labelProps = {};
          if (isStepSkipped(index)) {
            stepProps.completed = false;
          }
          return (
            <Step key={label} {...stepProps}>
              <StepLabel {...labelProps}></StepLabel>
            </Step>
          );
        })}
      </Stepper>
      {activeStep === steps.length ? (
        <React.Fragment>
          <Typography sx={{ mt: 2, mb: 1 }}>
            All steps completed - you&apos;re finished
            {/* Here we add the async logic with the prediction */}
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button onClick={handleReset}>Reset</Button>
          </Box>
        </React.Fragment>
      ) : (
        <React.Fragment sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
          <Typography sx={{ mt: 2, mb: 1 , display: 'flex', width: '60%'}}>
            {/* Step {activeStep + 1} */}
            <FormContainer />
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
                <Button onClick={handleNext} sx={{ margin: "0px 20px" }}>
                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                </Button>
          </Box>
        </React.Fragment>
      )}
    </Box>
  );
}
