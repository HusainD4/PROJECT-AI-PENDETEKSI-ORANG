document.addEventListener("DOMContentLoaded", function () {
    const saveBtn = document.getElementById("saveLimits");
    const maxPeopleInput = document.getElementById("max_people");
    const maxWeightInput = document.getElementById("max_weight");
    const resultDiv = document.getElementById("result");
    const warningBox = document.getElementById("warningBox");
    const peopleDisplay = document.getElementById("peopleDisplay");
    const beepSound = document.getElementById("beepSound");
    const lastUpdateDiv = document.getElementById("lastUpdate");

    let overloadPreviously = false;

    // 🔁 Fungsi polling status dari server
    function fetchStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                const {
                    people_count = 0,
                    total_weight = 0.0,
                    face_count = 0,
                    status = "UNKNOWN",
                    max_people_limit = 0,
                    max_weight_limit = 0.0
                } = data;

                // 🧾 Tampilkan status deteksi
                // Tampilkan status deteksi dalam tabel
                resultDiv.innerHTML = `
                    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%; max-width: 400px; text-align: left;">
                    <thead>
                        <tr style="background-color:#f2f2f2;">
                        <th style="text-align: left;">Parameter</th>
                        <th style="text-align: right;">Nilai</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td style="text-align: left;">Jumlah Orang</td><td style="text-align: right;">${people_count}</td></tr>
                        <tr><td style="text-align: left;">Jumlah Wajah</td><td style="text-align: right;">${face_count}</td></tr>
                        <tr><td style="text-align: left;">Berat Total (kg)</td><td style="text-align: right;">${total_weight.toFixed(1)}</td></tr>
                        <tr><td style="text-align: left;">Status</td><td style="text-align: right;">${status}</td></tr>
                    </tbody>
                    </table>
                `;

                lastUpdateDiv.textContent = "🕒 Terakhir diperbarui: " + new Date().toLocaleTimeString();

                // 🚨 Tampilkan peringatan jika overload
                // Logic suara dan peringatan
                const reachedPeopleLimit = (people_count >= max_people_limit && max_people_limit > 0);
                const reachedWeightLimit = (total_weight >= max_weight_limit && max_weight_limit > 0);

                if (status === "OVERLOAD") {
                    warningBox.style.display = "block";
                    if (!overloadPreviously) {
                        beepSound.play();
                        overloadPreviously = true;
                    }
                } else if (reachedPeopleLimit || reachedWeightLimit) {
                    // Jika sudah mencapai batas tapi belum overload
                    warningBox.style.display = "block";
                    if (!overloadPreviously) {
                        beepSound.play();
                        overloadPreviously = true;
                    }
                } else {
                    warningBox.style.display = "none";
                    if (overloadPreviously) {
                        beepSound.pause();
                        beepSound.currentTime = 0;
                        overloadPreviously = false;
                    }
                }


                // ℹ️ Info ringkas
                    // Info ringkas batas dalam tabel
                    peopleDisplay.innerHTML = `
                        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%; max-width: 300px; margin-top: 10px; text-align: left;">
                        <thead>
                            <tr style="background-color:#f9f9f9;">
                            <th style="text-align: left;">Batas Maksimal</th>
                            <th style="text-align: right;">Nilai</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td style="text-align: left;">Orang</td><td style="text-align: right;">${people_count} / ${max_people_limit}</td></tr>
                            <tr><td style="text-align: left;">Berat (kg)</td><td style="text-align: right;">${total_weight.toFixed(1)} / ${max_weight_limit}</td></tr>
                        </tbody>
                        </table>
                    `;

                // 🛠 Set nilai input batas jika belum pernah diset
                if (!maxPeopleInput.value) maxPeopleInput.value = max_people_limit;
                if (!maxWeightInput.value) maxWeightInput.value = max_weight_limit;

            })
            .catch(err => {
                console.error("Gagal memuat status:", err);
                resultDiv.textContent = "❌ Gagal memuat data dari server.";
                warningBox.style.display = "none";
                peopleDisplay.innerHTML = "";
                lastUpdateDiv.textContent = "⚠️ Gagal update";
            });
    }

    // 🔁 Polling status
    setInterval(fetchStatus, 1000);
    fetchStatus(); // panggilan awal

    // 💾 Simpan batas ke server
    saveBtn.addEventListener("click", () => {
    const people = parseInt(maxPeopleInput.value);
    const weight = parseFloat(maxWeightInput.value);

    if (isNaN(people) || isNaN(weight) || people <= 0 || weight <= 0) {
        Swal.fire("Input tidak valid", "Masukkan angka yang benar.", "warning");
        return;
    }

    fetch('/api/set_limits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_people: people, max_weight: weight })
    })
    .then(response => response.json())
    .then(data => {
        if(data.error) {
            Swal.fire("Gagal", data.error, "error");
        } else {
            Swal.fire("Sukses", "Batas maksimal diperbarui.", "success");
            fetchStatus(); // Refresh status segera
        }
    })
    .catch(error => {
        console.error("Gagal memperbarui batas:", error);
        Swal.fire("Gagal", "Tidak dapat menyimpan batas.", "error");
    });
});

});

