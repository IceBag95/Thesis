import { useRef, useState } from "react";

function Answers( { for_column, answers, currQuest, currAns, setCurrAns } ) {

    const selectedValue = useRef();
    const [invalidAnswer, setInvalidAnswer] = useState(false); 
    if (Array.isArray(answers) && answers.length > 0 && Object.keys(currAns).length === 0 ) {
        let newSelectedValue = answers[0].actual_value;
        selectedValue.current = newSelectedValue;
        setCurrAns({
                        for_column: for_column,
                        current_question: currQuest,
                        current_answer: newSelectedValue
                    });
    }
    else if (Array.isArray(answers) && answers.length == 0 && Object.keys(currAns).length === 0) {
        setCurrAns({
            for_column: for_column,
            current_question: currQuest,
            current_answer: null
        });
    }
    else {
        selectedValue.current = currAns.current_answer;
    }

    const handleRadioChange = (event) => {
        selectedValue.current = event.target.value;
        
        setCurrAns({
            for_column: for_column,
            current_question: currQuest,
            current_answer: event.target.value
        });
        
    }

    const handleNumberChange = (event) => {
        if (event.target.value >= 0){
            setInvalidAnswer(false);
            setCurrAns({
                for_column: for_column,
                current_question: currQuest,
                current_answer: event.target.value
            });
        }
        else {
            setInvalidAnswer(true);
        }
    }

    return(
        <div>
            {
                Array.isArray(answers) && answers.length > 0 ? (
                    answers.map((ans,idx) => {
                                return (
                                    <div>
                                            <input  type="radio" 
                                                    className="radio-input" 
                                                    id={`radiobutton_${idx}`} 
                                                    name="radiobutton" 
                                                    onChange={handleRadioChange} 
                                                    value={ans.actual_value} 
                                                    checked={ans.actual_value == selectedValue.current} />
                                            
                                            <label  for={`radiobutton_${idx}`}
                                                    className="radio-label">
                                                        {ans.shown_value}
                                            </label>
                                    </div>
                                )
                    })
                ) : Array.isArray(answers) && answers.length == 0 ? (
                    <>
                        <input  type="number" 
                                class="number-input"
                                value={currAns.current_answer || ""}
                                onChange={handleNumberChange}
                                style={invalidAnswer ? {backgroundColor: "rgb(245, 134, 132)", 
                                                        border: 'solid',
                                                        borderColor: "red", 
                                                        borderWidth: "2px",
                                                        borderRadius: "4px"} 
                                                    : {backgroundColor: "white", 
                                                        border: 'solid',
                                                        borderColor: "gray", 
                                                        borderWidth: "2px",
                                                        borderRadius: "4px"}} />
                        
                        {invalidAnswer ? 
                            <p style={{color: "red", fontSize: "0.7em"}}>Εισάγετε μόνο μη αρνητικές τιμές</p>
                        :   ""} 
                    </>
                ) : (
                    <p>Fetching Q&A</p>
                )
            }
        </div>

    )
}

export default Answers