import React from "react";
import Question from "./question";
import Answers from "./answers";
import '../Component-Styles/form_container.css'


function FormContainer() {

    return (
        <form className="form-container">
            <Question />
            <Answers />
        </form>
    )

}

export default FormContainer