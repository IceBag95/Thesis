import { useState, useEffect } from "react";
import Question from "./question";
import Answers from "./answers";
import '../Component-Styles/form_container.css'


function FormContainer( {step, currAns, setCurrAns, setNoQ, canGoNext, setCanGoNext} ) {
    const [initialData, setInitialData] = useState({});

    useEffect(() => {
        const getQnA = async () => {
            const response = await fetch("http://localhost:8000/get_initial_info");
            const data = await response.json()
            setInitialData(data);
            setNoQ(data.qna);
        }

        getQnA();
      }, [])

    return (
        <div className="form-container">
            <Question question={initialData.qna ? initialData.qna[step].question : null} />
            <Answers for_column={initialData.qna ? initialData.qna[step].for_column : null} 
                    answers={initialData.qna ? initialData.qna[step].answers : null} 
                    currQuest={initialData.qna ? initialData.qna[step].question : null} 
                    currAns={currAns}
                    setCurrAns={setCurrAns} 
                    currLimits={initialData.qna ? initialData.qna[step].limits : {}}
                    canGoNext={canGoNext}
                    setCanGoNext={setCanGoNext}/>
        </div>
    )

}

export default FormContainer