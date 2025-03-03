import React from "react";

function Answers( { data } ) {

    console.log(Array.isArray(data.answers));

    return(
        <ol>
            {Array.isArray(data.answers) ? (
                data.answers.map((ans, idx) => (
                    <li key={idx}>{ans}</li>
                ))
            ) : (
                <p>Trying to load answers</p>
            )}
        </ol>

    )
}

export default Answers