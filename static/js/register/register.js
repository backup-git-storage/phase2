"use strict";

document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("registerForm");

    const fullName = document.getElementById("fullName");
    const password = document.getElementById("password");
    const confirmPassword = document.getElementById("confirmPassword");

    const meterBar = document.getElementById("meterBar");
    const matchStatus = document.getElementById("matchStatus");


    // ------------------------------------------
    // PASSWORD SHOW / HIDE
    // ------------------------------------------

    document.querySelectorAll(".password-toggle").forEach((button) => {

        button.addEventListener("click", () => {

            const targetId = button.dataset.target;
            const input = document.getElementById(targetId);

            if (!input) {
                console.error("Password input not found:", targetId);
                return;
            }

            const passwordIsHidden = input.type === "password";

            input.type = passwordIsHidden ? "text" : "password";

            button.textContent = passwordIsHidden ? "Hide" : "Show";

            button.setAttribute(
                "aria-label",
                passwordIsHidden ? "Hide password" : "Show password"
            );

        });

    });


    // ------------------------------------------
    // CAPITALIZE NAME INITIALS
    // ------------------------------------------

    function capitalizeNameWords(value) {

        return value
            .toLocaleLowerCase()
            .replace(
                /(^|[\s'-])(\p{L})/gu,
                (match, separator, initial) => {
                    return separator + initial.toLocaleUpperCase();
                }
            );

    }


    if (fullName) {

        fullName.addEventListener("input", () => {

            const cursorPosition = fullName.selectionStart;

            const formatted = capitalizeNameWords(fullName.value);

            if (formatted !== fullName.value) {

                fullName.value = formatted;

                if (cursorPosition !== null) {
                    fullName.setSelectionRange(
                        cursorPosition,
                        cursorPosition
                    );
                }

            }

        });

        fullName.addEventListener("blur", () => {

            fullName.value = capitalizeNameWords(fullName.value)
                .trim()
                .replace(/\s+/g, " ");

        });

    }


    // ------------------------------------------
    // PASSWORD STRENGTH
    // ------------------------------------------

    function updatePasswordStrength() {

        if (!password || !meterBar) {
            return;
        }

        let score = 0;

        if (password.value.length >= 8) score++;
        if (/[A-Z]/.test(password.value)) score++;
        if (/[0-9]/.test(password.value)) score++;
        if (/[^A-Za-z0-9]/.test(password.value)) score++;

        meterBar.style.width = `${score * 25}%`;

        meterBar.style.background =
            score < 2 ? "#9e1b32" :
            score < 4 ? "#c56b78" :
            "#7eb78f";

    }


    // ------------------------------------------
    // PASSWORD RULES (letter + number + special character)
    // ------------------------------------------

    function checkPasswordRules() {

        if (!password) {
            return;
        }

        const value = password.value;

        if (!value) {
            password.setCustomValidity("");
            return;
        }

        if (!/[A-Za-z]/.test(value)) {
            password.setCustomValidity(
                "Password must contain at least one letter."
            );
        } else if (!/[0-9]/.test(value)) {
            password.setCustomValidity(
                "Password must contain at least one number."
            );
        } else if (!/[^A-Za-z0-9]/.test(value)) {
            password.setCustomValidity(
                "Password must contain at least one special character."
            );
        } else {
            password.setCustomValidity("");
        }

    }


    // ------------------------------------------
    // CONFIRM PASSWORD
    // ------------------------------------------

    function checkPasswordMatch() {

        if (!password || !confirmPassword || !matchStatus) {
            return;
        }

        if (!confirmPassword.value) {

            matchStatus.textContent = "";

            confirmPassword.setCustomValidity("");

            return;
        }

        if (password.value === confirmPassword.value) {

            matchStatus.textContent = "✓ Passwords match";

            matchStatus.style.color = "#8fc59d";

            confirmPassword.setCustomValidity("");

        } else {

            matchStatus.textContent = "Passwords do not match";

            matchStatus.style.color = "#e88998";

            confirmPassword.setCustomValidity(
                "Passwords do not match."
            );

        }

    }


    if (password) {
        password.addEventListener("input", () => {
            checkPasswordRules();
            updatePasswordStrength();
            checkPasswordMatch();
        });
    }

    if (confirmPassword) {
        confirmPassword.addEventListener(
            "input",
            checkPasswordMatch
        );
    }


    // ------------------------------------------
    // FINAL VALIDATION BEFORE SUBMISSION
    // ------------------------------------------

    if (form) {

        form.addEventListener("submit", (event) => {

            if (fullName) {

                fullName.value = capitalizeNameWords(fullName.value)
                    .trim()
                    .replace(/\s+/g, " ");

            }

            checkPasswordRules();
            checkPasswordMatch();

            if (!form.checkValidity()) {

                event.preventDefault();

                form.reportValidity();

            }

        });

    }

});