document.addEventListener("DOMContentLoaded", () => {
    const card = document.getElementById('tilt-card');
    const container = document.querySelector('.container');
    const imageInput = document.getElementById('imageInput');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');

    // 1. 3D Dynamic Tilt
    container.addEventListener('mousemove', (e) => {
        let xAxis = (window.innerWidth / 2 - e.pageX) / 35;
        let yAxis = (window.innerHeight / 2 - e.pageY) / 35;
        card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    });

    container.addEventListener('mouseleave', () => {
        card.style.transition = "all 0.5s ease";
        card.style.transform = `rotateY(0deg) rotateX(0deg)`;
    });

    container.addEventListener('mouseenter', () => {
        card.style.transition = "none";
    });

    // 2. Update File Name
    imageInput.addEventListener('change', (e) => {
        const name = e.target.files[0] ? e.target.files[0].name : "LOAD SUBMERGED DATA";
        fileNameDisplay.textContent = name.toUpperCase();
        fileNameDisplay.style.color = "#00f2ff";
    });

    // 3. Process Submission
    uploadForm.addEventListener("submit", async function(e) {
        e.preventDefault();

        // Reveal Loader
        loader.style.display = "block";
        document.getElementById("resultsSection").style.display = "none";
        document.getElementById("metricsSection").style.display = "none";

        const formData = new FormData(this);

        try {
            const response = await fetch("/process", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            // Simulate slight delay for "Processing" feel
            setTimeout(() => {
                loader.style.display = "none";

                // Show Results
                document.getElementById("resultsSection").style.display = "flex";
                document.getElementById("metricsSection").style.display = "block";

                // Map Images
                document.getElementById("originalImage").src = data.original;
                document.getElementById("enhancedImage").src = data.enhanced;
                document.getElementById("metricsData").innerText = data.metrics;
                document.getElementById("downloadBtn").href = data.enhanced;
                
                // Scroll to results
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            }, 1000);

        } catch (error) {
            console.error("Enhancement failed:", error);
            loader.innerHTML = "<p style='color:red'>DATA LOSS: RECHECK UPLINK</p>";
        }
    });
});