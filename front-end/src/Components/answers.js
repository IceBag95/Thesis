import React, { useRef } from "react";

function Answers( { for_column, answers, currQuest, currAns, setCurrAns } ) {

    const selectedValue = useRef();
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
        setCurrAns({
            for_column: for_column,
            current_question: currQuest,
            current_answer: event.target.value
        });
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
                    <input  type="number" 
                            class="number-input"
                            value={currAns.current_answer || ""}
                            onChange={handleNumberChange} />
                ) : (
                    <p>Trying to load answers</p>
                )
            }
        </div>

    )
}

export default Answers