"use strict";

document.addEventListener("DOMContentLoaded", () => {

    const passwordButtons = document.querySelectorAll(".password-toggle");

    passwordButtons.forEach((button) => {

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

});