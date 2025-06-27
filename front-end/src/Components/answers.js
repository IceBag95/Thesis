import { useRef } from "react";

function Answers( { for_column, answers, currQuest, currAns, setCurrAns, currLimits, canGoNext, setCanGoNext } ) {

    console.log('rendering answers')
    const selectedValue = useRef();
    if (Array.isArray(answers) && answers.length > 0 && Object.keys(currAns).length === 0) {
        console.log('entered if');
        let newSelectedValue = answers[0].actual_value;
        selectedValue.current = newSelectedValue;
//        setCanGoNext(true);
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
        if (currAns.current_answer) {
            setCanGoNext(true)
        }
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
        const ans = event.target.value

        if (ans >= currLimits.lower_limit && ans <= currLimits.upper_limit){
            setCanGoNext(true);
        }
        else {
            setCanGoNext(false);
        }
        setCurrAns({
            for_column: for_column,
            current_question: currQuest,
            current_answer: ans
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
                    <>
                        <input  type="number" 
                                class="number-input"
                                value={currAns.current_answer || ""}
                                onChange={handleNumberChange}
                                style={!canGoNext? {
                                                        backgroundColor: "rgb(245, 134, 132)", 
                                                        border: 'solid',
                                                        borderColor: "red", 
                                                        borderWidth: "2px",
                                                        borderRadius: "4px",
                                                        color:"red"} 
                                                    : {backgroundColor: "white", 
                                                        border: 'solid',
                                                        borderColor: "gray", 
                                                        borderWidth: "2px",
                                                        borderRadius: "4px",
                                                        color: "black"}} />
                        
                        {!canGoNext ? 
                            <p style={{color: "red", fontSize: "0.7em"}}>Εισάγετε τιμές μεταξύ {currLimits.lower_limit} και {currLimits.upper_limit}</p>
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