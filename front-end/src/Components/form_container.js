import React, { useState, useEffect } from "react";
import Question from "./question";
import Answers from "./answers";
import '../Component-Styles/form_container.css'


function FormContainer() {
    const [initialData, setInitialData] = useState({});

    useEffect(() => {
        const getQnA = async () => {
            const response = await fetch("http://localhost:8000/get_initial_info");
            const data = await response.json()
            console.log('data:', data)
            setInitialData(data);

            console.log(Array.isArray(data.answers));
        }

        getQnA();
      }, [])

    return (
        <div className="form-container">
            <Question question={initialData.question} />
            <Answers data={initialData} />
        </div>
    )

}

export default FormContainer