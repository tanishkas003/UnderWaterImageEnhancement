document.addEventListener("DOMContentLoaded", () => {
    const card = document.getElementById('tilt-card');
    const container = document.querySelector('.container');
    const imageInput = document.getElementById('imageInput');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');

    // ===============================
    // 1. 3D Dynamic Tilt
    // ===============================
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

    // ===============================
    // 2. Update File Name
    // ===============================
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        const name = file ? file.name : "LOAD SUBMERGED DATA";
        fileNameDisplay.textContent = name.toUpperCase();
        fileNameDisplay.style.color = "#00f2ff";
    });

    // ===============================
    // 3. Process Submission (FIXED)
    // ===============================
    uploadForm.addEventListener("submit", async function(e) {
        e.preventDefault();

        loader.style.display = "block";
        loader.innerHTML = "Processing underwater data...";

        document.getElementById("resultsSection").style.display = "none";
        document.getElementById("metricsSection").style.display = "none";

        const file = imageInput.files[0];

        // ✅ Prevent empty submission
        if (!file) {
            alert("Please select an image first.");
            loader.style.display = "none";
            return;
        }

        // ✅ Explicit FormData (more reliable)
        const formData = new FormData();
        formData.append("image", file);

        const mode = document.getElementById("modeSelect").value;
        formData.append("mode", mode);

        try {
            const response = await fetch("/enhance", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Server error");
            }

            loader.style.display = "none";

            // ===============================
            // Show Results
            // ===============================
            document.getElementById("resultsSection").style.display = "flex";
            document.getElementById("metricsSection").style.display = "block";

            document.getElementById("originalImage").src =
                data.original_image_url + "?t=" + new Date().getTime();

            document.getElementById("enhancedImage").src =
                data.enhanced_image_url + "?t=" + new Date().getTime();

            document.getElementById("metricsData").innerHTML =
                `<strong>PSNR:</strong> ${data.psnr} dB<br>
                 <strong>Entropy:</strong> ${data.entropy}`;

            document.getElementById("downloadBtn").href =
                data.download_url;

            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });

        } catch (error) {
            console.error("Enhancement failed:", error);
            loader.innerHTML =
                "<p style='color:red'>DATA LOSS: RECHECK UPLINK</p>";
        }
    });
});
